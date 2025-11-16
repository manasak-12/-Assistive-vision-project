# from modules.navigation.navigation_worker import NavigationWorker
# from modules.events.event_manager import EventManager
# from modules.speech.speech_worker import SpeechWorker
# import time

# # ----------------------------------------------------------
# # CONFIGURATION — Set start & destination coordinates
# # (You can change these later based on your area)
# # ----------------------------------------------------------
# START = (12.9352, 77.5340)  # Example: RR Nagar, Bangalore
# END   = (12.9345, 77.5365)

# def main():
#     # Initialize modules
#     navigator = NavigationWorker(start_point=START, end_point=END)
#     events = EventManager()
#     speech = SpeechWorker(vosk_model_path="models/vosk_small_model")

#     # Load route
#     navigator.load_route()

#     # Start Voice Command Listener (non-blocking)
#     speech.start_listening()

#     speech.speak("Navigation system is ready. Starting route guidance.")

#     # Main navigation loop
#     index = 0

#     while True:
#         current_location = navigator.simulate_gps()

#         if current_location is None:
#             speech.speak("You have reached your destination.")
#             break

#         print("Current GPS:", current_location)

#         # Generate event for this segment
#         instruction = events.generate_event(navigator.route_coords, index)
#         print("Instruction:", instruction)

#         # Speak the instruction
#         if "IN" in instruction or "TURN" in instruction:
#             speech.speak(instruction)

#         # Destination reached
#         if instruction == "DESTINATION REACHED":
#             speech.speak("Destination reached.")
#             break

#         index += 1
#         time.sleep(1)

# if __name__ == "__main__":
#     main()

from modules.navigation.navigation_worker import NavigationWorker
from modules.events.event_manager import EventManager
from modules.speech.speech_worker import SpeechWorker
from modules.events.event_bus import EventBus
import time

SIMULATION_MODE = True

START = (12.9352, 77.5340)
END   = (12.9345, 77.5365)

def main():
    event_bus = EventBus()
    event_bus.start()

    navigator = NavigationWorker(start_point=START, end_point=END)
    event_manager = EventManager()
    speech = SpeechWorker(vosk_model_path="models/vosk_small_model", event_bus=event_bus)

    navigator.load_route()
    speech.start_listening()
    speech.speak("Navigation system is ready.")

    # -------------------------------
    # EVENT HANDLERS
    # -------------------------------

    # SPEECH → Navigation
    def handle_voice_command(data):
        cmd = data["text"]
        print("[EVENT] Voice Command:", cmd)

        if "stop" in cmd:
            speech.speak("Stopping navigation.")
        elif "repeat" in cmd:
            speech.speak("Repeating instruction.")
        # future: user says “take me to St Peters”, vision OCR confirms, etc.

    event_bus.subscribe("voice_command", handle_voice_command)

    # NAVIGATION → Speech
    def handle_navigation_instruction(data):
        speech.speak(data["text"])

    event_bus.subscribe("navigation_instruction", handle_navigation_instruction)

    # -------------------------------
    # MAIN LOOP
    # -------------------------------
    index = 0
    print("\n=== Starting Navigation ===\n")

    while True:
        if SIMULATION_MODE:
            gps_point = navigator.simulate_gps()
        else:
            gps_point = navigator.get_current_gps()

        if gps_point is None:
            event_bus.publish({
                "type": "navigation_instruction",
                "data": {"text": "Destination reached."}
            })
            break

        print("Current GPS:", gps_point)

        instruction = event_manager.generate_event(navigator.route_coords, index)
        print("Instruction:", instruction)

        # publish navigation event
        event_bus.publish({
            "type": "navigation_instruction",
            "data": {"text": instruction}
        })

        if instruction == "DESTINATION REACHED":
            break

        index += 1
        time.sleep(1)

    print("\n=== Navigation Completed ===\n")

if __name__ == "__main__":
    main()
