"""Deliver exact expert speech through the existing session's sole socket writer.

The foreground model does not author, summarize or synthesize these answers.
This object owns media only, never the task that produced it.
"""
from __future__ import annotations

import asyncio
import base64
from contextlib import suppress

from app.attention.models import AttentionError
from app.attention.tts import SAMPLE_RATE, SpeechError
from app.realtime.events import encode_event


from app.attention.notifications import speech_attribution


class ExpertVoiceDelivery:
    def __init__(self, repository, speech, claim, session, *, receipt_seconds=60):
        self.repository, self.speech, self.claim, self.session = repository, speech, claim, session
        self.response_id = f"tts_{claim.attempt_id.hex}"
        self.receipt_seconds = receipt_seconds
        self.revoked = False
        self.reason = "delivery_failed"
        self.task = None
        self._played = asyncio.Event()
        self._items = {}
        self._clear_task = None
        self._started = False
        self.audio_origin, self.authored_by, _ = speech_attribution(claim.source_namespace)

    def item_id(self, index):
        return f"{self.response_id}_{index}"

    def current(self):
        return (not self.revoked and not self.session._closed.is_set()
                and not getattr(self.session, "_context_revoked", False)
                and self.session._attention_delivery is self)

    async def is_current(self):
        # The network generator's local cancellation check is not a second
        # database round trip for every PCM packet. Exact source authorization
        # remains mandatory at submission and under EVERY actual phone write.
        return self.current()

    async def _send(self, index, event_type, *, start_sample=None, samples=0, **payload):
        event = dict(type=event_type, provider="openai", audio_origin=self.audio_origin, response_id=self.response_id,
                     item_id=self.item_id(index), content_index=0, **payload)
        async with asyncio.timeout(5):
            sent = await self.session._app_sender.send_text(
                encode_event(event.pop("type"), **event), should_send=self.current,
                write_guard=lambda: self.repository.write_guard(
                    self.claim, index, start_sample=start_sample, samples=samples))
        if not sent:
            raise SpeechError("tts_phone_not_written")

    def _capture(self, index, *, complete=False, interrupted=False):
        if capture := self.session._context_capture:
            item = self._items[index]
            capture.tts_speech(
                response_id=self.response_id, item_id=self.item_id(index), text=item["text"],
                samples=max(item["samples"], item.get("played_samples", 0)), complete=complete, interrupted=interrupted,
                source_payload_id=self.claim.source_payload_id, authored_by=self.authored_by,
                provenance={"intent_id":str(self.claim.intent_id), "attempt_id":str(self.claim.attempt_id),
                            "delivery_generation":self.claim.generation, "chunk_index":index})

    async def _heartbeat(self):
        while self.current() and not self._played.is_set():
            await asyncio.sleep(self.repository.lease_seconds / 3)
            try:
                await self.repository.renew(self.claim)
            except AttentionError:
                # A completion receipt may have terminalized the attempt while
                # this heartbeat was waiting for its transaction.
                if not self._played.is_set():
                    await self.stop("delivery_source_revoked")
                return

    def _playback_timeout_seconds(self):
        # Receipts are cumulative per chunk. Budget the actual unacknowledged
        # PCM duration, not the time already spent generating earlier chunks.
        queued_samples = sum(max(0, item["samples"] - item.get("played_samples", 0))
                             for item in self._items.values())
        return self.receipt_seconds + queued_samples / SAMPLE_RATE

    async def run(self):
        self._started = True
        heartbeat = asyncio.create_task(self._heartbeat())
        try:
            async with asyncio.timeout(300):
                if not self.current():
                    raise SpeechError("tts_source_revoked")
                requests = await self.repository.prepare(self.claim, self.speech)
            for index, request in enumerate(requests):
                # Bound each prepared chunk independently. An aggregate wall
                # clock limit would truncate healthy long, multi-chunk speech.
                async with asyncio.timeout(300):
                    if not self.current():
                        raise SpeechError("tts_source_revoked")
                    item = self._items[index] = {"text":request.text, "samples":0, "complete":False}
                    self.session.remember_attention_item(self.item_id(index), self)

                    async def authorize(exact_request):
                        if not self.current():
                            return False
                        return await self.repository.submit_chunk(self.claim, index, exact_request)

                    async with self.speech.open_stream(request, authorize_submission=authorize,
                                                       is_current=self.is_current) as stream:
                        async for frame in stream.frames():
                            await self._send(index, "audio.output.delta", start_sample=frame.start_sample,
                                samples=frame.end_sample-frame.start_sample,
                                audio_base64=base64.b64encode(frame.data).decode("ascii"))
                            item["samples"] = frame.end_sample
                            self._capture(index)
                        await self.repository.chunk_generated(self.claim, index, samples=stream.samples_yielded,
                            first_bytes_ms=stream.first_bytes_ms, provider_request_id=stream.provider_request_id)
                    item["complete"] = True
                    self._capture(index, complete=True)
                    await self._send(index, "transcript.done", role="assistant", text=request.text)
                    await self.session.record_attention_speech(self, index, request.text)
                    if index == len(requests)-1:
                        # Persist completion before exposing the final done event:
                        # a very fast client receipt must not beat this transition.
                        await self.repository.generation_finished(self.claim)
                        await self._send(index, "response.done", status="completed")
                    await self._send(index, "audio.output.done")
            # Generation is not playback. Keep the voice lane occupied until
            # a validated receipt, barge-in, disconnection or explicit clear.
            # This fixed deadline allows the queued tail to drain, but does
            # not grow indefinitely when receipts are absent or repeated.
            async with asyncio.timeout(self._playback_timeout_seconds()):
                await self._played.wait()
        except asyncio.CancelledError:
            self.reason = self.reason if self.revoked else "delivery_cancelled"
        except TimeoutError:
            self.reason = "playback_or_generation_timeout"
        except (AttentionError, SpeechError) as error:
            if not self.revoked:
                self.reason = error.code
        except Exception:
            # Provider/database exception strings may contain private content.
            self.reason = "delivery_failed"
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            if not self._played.is_set():
                self.revoked = True
                for index, item in self._items.items():
                    self._capture(index, complete=item["complete"], interrupted=True)
                await self._clear()
                with suppress(Exception):
                    await self.repository.settle(self.claim, reason=self.reason,
                                                 interrupted=self.reason == "user_interrupted")
            await self.session.release_attention(self)

    async def _clear(self):
        if self.session._closed.is_set():
            return
        if self._clear_task is None:
            self._clear_task = asyncio.create_task(self._send_clear())
        # Heartbeat revocation can cancel the producer while both are trying
        # to clear. One cancellation must not abandon their shared clear write.
        await asyncio.shield(self._clear_task)

    async def _send_clear(self):
        # This is a revocation/control message, so it must not require the
        # now-revoked content source to remain available.
        try:
            async with asyncio.timeout(2):
                for index in self._items:
                    await self.session._app_sender.send_text(encode_event("audio.output.clear",
                        provider="openai", audio_origin=self.audio_origin, response_id=self.response_id, item_id=self.item_id(index),
                        reason=self.reason))
        except Exception:
            # Do not start another utterance if stopping this one was uncertain.
            self.session._mark_closing("attention_clear_unknown")

    async def stop(self, reason):
        self.revoked, self.reason = True, reason
        if self._started and self.task is not None and self.task is not asyncio.current_task() and not self.task.done():
            self.task.cancel()
        await self._clear()

    async def receipt(self, event, *, completed):
        item_id = event.get("item_id")
        if item_id not in {self.item_id(index) for index in self._items}:
            raise AttentionError("attention_playback_invalid", "Unknown speech item.")
        response_id = event.get("response_id", self.response_id if not completed else None)
        if type(event.get("content_index", 0)) is not int or event.get("content_index", 0) != 0:
            raise AttentionError("attention_playback_invalid", "Invalid content index.")
        if event.get("type") != "playback.truncated":
            samples = event.get("played_samples")
        else:
            milliseconds = event.get("audio_end_ms", 0)
            if type(milliseconds) is not int or milliseconds < 0:
                raise AttentionError("attention_playback_invalid", "Invalid playback time.")
            samples = milliseconds * 24
        all_played = await self.repository.playback_receipt(self.claim, item_id=item_id,
            response_id=response_id, samples=samples, completed=completed)
        index = next(i for i in self._items if self.item_id(i) == item_id)
        item = self._items[index]
        item["played_samples"] = max(item.get("played_samples", 0), samples)
        if capture := self.session._context_capture:
            # A fast client can acknowledge a committed socket write before
            # the producer resumes its local capture callback. The durable
            # receipt has already bounded these samples by actual forwarding.
            self._capture(index, complete=self._items[index]["complete"], interrupted=self.revoked)
            capture.playback_receipt(item_id, response_id=self.response_id, samples=samples,
                state="interrupted" if event.get("type") == "playback.truncated" else event["type"].removeprefix("playback."))
        if all_played:
            self._played.set()
        if event.get("type") in {"playback.truncated", "playback.interrupted"} and self.current():
            await self.stop("user_interrupted")
