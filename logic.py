from typing import List
from enum import Enum
import re

from vision import Detection, build_scene_description


class CommandType(Enum):
    NONE = "none"
    DESCRIBE_SCENE = "describe_scene"
    READ_TEXT = "read_text"
    WHO_IS_HERE = "who_is_here"
    REPEAT_LAST = "repeat_last"
    QUIET_MODE = "quiet_mode"          # stop auto speech
    START_TALKING = "start_talking"    # enable auto speech
    HOW_ARE_YOU = "how_are_you"
    WEATHER = "weather"
    DESCRIBE_EMOTION = "describe_emotion"
    COUNT_OBJECTS = "count_objects"
    STOP_LISTENING = "stop_listening"
    START_LISTENING = "start_listening"
    STOP_DETECTION = "stop_detection"
    START_DETECTION = "start_detection"
    HELLO = "hello"


def decide_scene_speech(detections: List[Detection]) -> str:
    """
    Build a sentence describing the scene based on detections.
    """
    return build_scene_description(detections)


def _normalize(text: str) -> str:
    """
    Lowercase and remove punctuation so small mistakes don't break matching.
    """
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def interpret_voice_command(text: str) -> CommandType:
    """
    Map raw recognized text to a command.
    Uses BOTH phrase checks and word-level checks so that
    small recognition mistakes (like 'listing' vs 'listening')
    still work.
    """
    t = _normalize(text)
    words = t.split()
    word_set = set(words)

    # Helper: check if any of the given phrases appear
    def has_any(phrases):
        return any(p in t for p in phrases)

    # Helper: check if all tokens are present as separate words
    def has_words(*needed):
        return all(any(needed_word in w for w in words) for needed_word in needed)

    # -------- Scene / object description --------
    if has_any([
        "describe the scene",
        "describe scene",
        "describe this scene",
        "what do you see",
        "what do u see",
        "detect the object",
        "detect object",
        "detect objects",
        "tell me what you see",
        "tell me what u see",
        "whats in front of me",
        "what is in front of me",
        "what is the scenery",
        "whats the scenery",
        "what is the scene",
        "whats the scene",
        "tell me the scenery",
        "tell me the scene",
        "can you tell me the scenery",
        "can u tell me the scenery",
        "can you describe the scenery",
        "can u describe the scenery",
    ]):
        return CommandType.DESCRIBE_SCENE

    # Fallback: scene/scenery word + question words
    if ("scene" in t or "scenery" in t or "sceneery" in t) and (
        "what" in t or "tell" in t or "describe" in t
    ):
        return CommandType.DESCRIBE_SCENE

    # -------- Text / board reading --------
    if has_any([
        "read text",
        "read the text",
        "read board",
        "read the board",
        "read sign",
        "read the sign",
        "read what is written",
        "read whats written",
        "read the words",
    ]):
        return CommandType.READ_TEXT

    # -------- Who is here / in front --------
    if has_any([
        "who is here",
        "who is there",
        "who is this",
        "whos here",
        "whos this",
        "who is in front of me",
        "who is in front of you",
        "who is in front of u",
        "whos in front of me",
        "whos in front of you",
        "whos in front of u",
        "who is in front",
        "whos in front",
        "who stands in front of me",
        "who is standing in front of me",
        "who is standing in front",
        "recognise the person",
        "recognize the person",
    ]):
        return CommandType.WHO_IS_HERE

    # -------- Repeat last --------
    if has_any([
        "repeat that",
        "say again",
        "repeat it",
        "repeat what you said",
        "say that again",
    ]):
        return CommandType.REPEAT_LAST

    # -------- Quiet / stop talking --------
    if has_any([
        "stop talking",
        "be quiet",
        "quiet mode",
        "stop voice",
        "stop speaking",
        "dont talk",
        "don't talk",
        "no talking",
        "stop your voice",
    ]):
        return CommandType.QUIET_MODE

    # -------- Start talking / speak mode --------
    if has_any([
        "start talking",
        "speak mode",
        "start speaking",
        "you can talk",
        "talk now",
        "start voice",
    ]):
        return CommandType.START_TALKING

    # -------- Stop / start listening --------
    # we handle common Vosk mistakes: "listening", "listin", "listing"
    if (
        "stop listening" in t
        or has_words("stop", "listen")
        or has_words("stop", "listening")
        or has_words("stop", "listing")
    ):
        return CommandType.STOP_LISTENING

    if (
        "start listening" in t
        or has_words("start", "listen")
        or has_words("start", "listening")
        or has_words("start", "listing")
        or "you can listen" in t
    ):
        return CommandType.START_LISTENING

    # -------- Count objects --------
    # accept "count objects", "count the objects", "count object"
    if (
        "count objects" in t
        or "count the objects" in t
        or "count object" in t
        or has_words("count", "object")
        or has_words("count", "objects")
    ):
        return CommandType.COUNT_OBJECTS

    # -------- Stop / start detection --------
    if "stop detection" in t:
        return CommandType.STOP_DETECTION
    if "start detection" in t or "resume detection" in t:
        return CommandType.START_DETECTION

    # -------- Small talk / hello --------
    if any(w in word_set for w in ["hello", "hi", "hey"]):
        return CommandType.HELLO

    if "how are you" in t or "how r u" in t:
        return CommandType.HOW_ARE_YOU

    # -------- Weather (offline) --------
    if has_any([
        "how is the weather",
        "hows the weather",
        "whats the weather",
        "what is the weather",
    ]):
        return CommandType.WEATHER

    # -------- Describe emotion --------
    if has_any([
        "describe the emotion",
        "describe emotion",
        "what is the emotion",
        "whats the emotion",
        "whats their emotion",
    ]):
        return CommandType.DESCRIBE_EMOTION

    return CommandType.NONE
