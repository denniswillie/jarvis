"""Offline WebRTC tone through the real voice-smoke capture harness.

No model calls, microphone or audible speaker output. Uses two local peers.
"""
import argparse
import asyncio
import json
from pathlib import Path

from evals.responsive_voice.synthetic_webrtc import Browser
from live_hermes_voice_acceptance import BROWSER_PROGRAM


TONE = r"""(async()=>{
 const offer=await acceptance.prepare();
 const ac=new AudioContext({sampleRate:48000});await ac.resume();
 const other=new RTCPeerConnection(),tone=ac.createOscillator(),gain=ac.createGain();
 const output=ac.createMediaStreamDestination();gain.gain.value=.2;
 tone.connect(gain);gain.connect(output);
 try {
   output.stream.getTracks().forEach(t=>other.addTrack(t,output.stream));
   await other.setRemoteDescription({type:'offer',sdp:offer.sdp});
   await other.setLocalDescription(await other.createAnswer());
   if(other.iceGatheringState!=='complete')await new Promise((resolve,reject)=>{
     const timer=setTimeout(()=>reject(Error('local_ice_timeout')),10000);
     other.addEventListener('icegatheringstatechange',()=>{
       if(other.iceGatheringState==='complete'){clearTimeout(timer);resolve();}});});
   await acceptance.answer(other.localDescription.sdp);acceptance.activate();tone.start();
   await new Promise(r=>setTimeout(r,900));return acceptance.snapshot();
 } finally {other.close();output.stream.getTracks().forEach(t=>t.stop());await ac.close();}
})()"""


async def run(root, wav):
    root.mkdir(mode=0o700)
    browser = Browser(root, wav, disable_mdns=True)
    report = {"scope": "offline_local_webrtc_tone", "physical_playback": False}
    try:
        await browser.start()
        await browser.cdp.evaluate(BROWSER_PROGRAM)
        async with asyncio.timeout(30):
            report["audio"] = await browser.cdp.evaluate(TONE)
        report["passed"] = (report["audio"].get("firstSignal") is not None
                            and report["audio"].get("fatal") is None)
    finally:
        report["cleanup"] = await browser.close()
        (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("wav", type=Path)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.root, args.wav)))
