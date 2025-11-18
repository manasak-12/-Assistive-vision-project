import queue
import threading

class EventBus:
    def __init__(self):
        self.event_queue = queue.Queue()
        self.subscribers = {}  # { "event_type": [callback1, callback2] }

    # PUBLISH EVENT
    def publish(self, event):
        """
        event structure:
        {
            "type": "voice_command" / "vision" / "navigation_instruction"
            "data": {...}
        }
        """
        self.event_queue.put(event)

    # SUBSCRIBE
    def subscribe(self, event_type, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    # START EVENT LOOP
    def start(self):
        threading.Thread(target=self._event_loop, daemon=True).start()

    # INTERNAL LOOP
    def _event_loop(self):
        while True:
            event = self.event_queue.get()
            event_type = event["type"]

            if event_type in self.subscribers:
                for cb in self.subscribers[event_type]:
                    cb(event["data"])