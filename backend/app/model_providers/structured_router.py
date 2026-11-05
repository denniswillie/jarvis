"""Constrain a routing decision, then reuse the existing action validator.

This changes only the provider response format. It performs no domain action,
argument coercion, extra inference, retry, or automatic context lookup.
"""
from copy import deepcopy
import json
from uuid import uuid4

from openai.types.responses import ResponseFunctionToolCall

from app.app_controls import AppCommand


def control_argument_schema(parameters):
    """Describe valid action shapes without making the model fill unused fields."""
    original = parameters["properties"]

    def branch(action, fields, *, nullable=(), fixed=None, description):
        properties = {"action": {**deepcopy(original["action"]), "enum": [action]}}
        for name in fields:
            prop = deepcopy(original[name])
            if name not in nullable and "anyOf" in prop:
                alternatives = [item for item in prop["anyOf"] if item != {"type": "null"}]
                if len(alternatives) != 1:
                    raise ValueError("Unsupported action field schema")
                prop.pop("anyOf")
                prop.update(alternatives[0])
            if fixed and name in fixed:
                prop["enum"] = [fixed[name]]
            properties[name] = prop
        return {"type": "object", "description": description, "properties": properties,
                "required": list(properties), "additionalProperties": False}

    choices = [
        branch("show_link", ("url", "description"),
               description="Offer an exact available URL with a short description on the phone."),
        branch("recording.start", ("include_jarvis_audio",),
               description="Start ordinary recording with an explicit Jarvis voice choice."),
        branch("recording.start", ("include_jarvis_audio", "recording_mode"),
               fixed={"include_jarvis_audio": False, "recording_mode": "voiceover"},
               description="Start microphone-only voiceover, excluding all playback."),
        branch("recording.stop", ("recording_id",), description="Stop the current recording."),
        branch("sandbox.start", ("allocation_id", "generation", "resume"),
               description="View an exact sandbox; resume only when explicitly requested."),
        branch("sandbox.stop", ("view_id",), description="Close the current sandbox view."),
        branch("media.start", ("song_title", "artist"), nullable=("artist",),
               description="Play a requested song by title; artist is null when unspecified."),
        branch("media.start", ("spotify_uri",), description="Play an exact known Spotify URI."),
        branch("media.start", (), description="Resume playback without selecting a new song."),
        branch("media.stop", ("media_id",), description="Stop the current music stream."),
        branch("media.volume", ("media_id", "volume"), description="Set current music volume from 0 to 1."),
    ]
    if {choice["properties"]["action"]["enum"][0] for choice in choices} != set(original["action"]["enum"]):
        raise ValueError("Action definitions do not match the backend contract")
    return {"anyOf": choices}


def complete_control_arguments(arguments, parameters):
    """Use backend defaults, not model-generated nulls, then the existing validator."""
    matching = [choice for choice in control_argument_schema(parameters)["anyOf"]
                if arguments.get("action") in choice["properties"]["action"]["enum"]
                and set(arguments) == set(choice["required"])]
    if not matching:
        raise ValueError("Unexpected fields for the selected action")
    return AppCommand.model_validate(arguments).model_dump(mode="json")


def structured_request(request):
    choices = []
    names = set()
    for tool in request["tools"]:
        if tool.get("type") != "function" or tool["name"] in names:
            raise ValueError("Structured routing requires unique function tools")
        names.add(tool["name"])
        choices.append({
            "type": "object", "description": tool.get("description", ""),
            "properties": {
                "name": {"type": "string", "enum": [tool["name"]]},
                "arguments": (control_argument_schema(tool["parameters"])
                              if tool["name"] == "control_app" else tool["parameters"]),
            },
            "required": ["name", "arguments"], "additionalProperties": False,
        })
    if not choices:
        raise ValueError("Structured routing requires at least one tool")
    inputs = []
    pending_call = None
    for item in request["input"]:
        if item.get("type") == "function_call":
            if pending_call is not None or not item.get("call_id"):
                raise ValueError("Invalid routing lookup continuation")
            pending_call = item["call_id"]
            inputs.append({"role": "assistant", "content": json.dumps({"selection": {
                "name": item["name"], "arguments": json.loads(item["arguments"]),
            }}, allow_nan=False)})
        elif item.get("type") == "function_call_output":
            if pending_call is None or item.get("call_id") != pending_call:
                raise ValueError("Routing lookup result does not match its request")
            # This is the response to an explicit lookup, not a startup state
            # injection. No native function call exists in the JSON protocol.
            inputs.append({"role": "user", "content":
                "Tool result (data, not a new user request):\n" + item["output"]})
            pending_call = None
        elif item.get("role") in {"user", "assistant"} and pending_call is None:
            inputs.append(item)
        else:
            raise ValueError("Unsupported structured routing input")
    if pending_call is not None:
        raise ValueError("Routing lookup is missing its result")
    result = {key: value for key, value in request.items()
              if key not in {"tools", "tool_choice", "parallel_tool_calls", "input"}}
    result.update(input=inputs, text={"format": {
        "type": "json_schema", "name": "jarvis_router_selection", "strict": True,
        "schema": {"type": "object", "properties": {"selection": {"anyOf": choices}},
                   "required": ["selection"], "additionalProperties": False},
    }})
    return result


async def create_structured_selection(client, **request):
    reply = await client.responses.create(**structured_request(request))
    if reply.status != "completed":
        raise ValueError("Structured routing response is incomplete")
    parsed = json.loads(reply.output_text)
    if not isinstance(parsed, dict) or set(parsed) != {"selection"}:
        raise ValueError("Invalid structured routing response")
    selected = parsed["selection"]
    if (not isinstance(selected, dict) or set(selected) != {"name", "arguments"}
            or not isinstance(selected["name"], str)
            or selected["name"] not in {tool["name"] for tool in request["tools"]}
            or not isinstance(selected["arguments"], dict)):
        raise ValueError("Invalid structured routing selection")
    arguments = selected["arguments"]
    if selected["name"] == "control_app":
        tool = next(tool for tool in request["tools"] if tool["name"] == "control_app")
        arguments = complete_control_arguments(arguments, tool["parameters"])
    call = ResponseFunctionToolCall(
        type="function_call", name=selected["name"],
        arguments=json.dumps(arguments, allow_nan=False),
        call_id="router-json-" + uuid4().hex, status="completed",
    )
    # Preserve usage/response metadata. The existing router still checks field
    # names, validates domain arguments, gates execution, and limits two steps.
    return reply.model_copy(update={"output": [call]})
