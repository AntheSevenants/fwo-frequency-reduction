import json
import urllib.request

from typing import Union, Dict, Any
from dataclasses import dataclass


class EventState:
    """Collection of events that can happen during a batch run."""

    PROGRESS: int = 0
    FINISHED: int = 1


class EventType:
    """Types of events that can happen during a batch run."""

    BATCH_RUN: int = 0
    AGGREGATION: int = 1


@dataclass
class Event:
    """Description of an event as it happens during a batch run."""

    selected_sweep: str
    state: int
    etype: int
    progress: Union[int, float, None] = None


class Webhook:
    """Configuration for a webhook callback that fires during batch run events."""

    def __init__(self, url: str, api_key: str, payload: Dict[str, Any]):
        """Initialised the webhook configuration.

        Args:
            url (str): the endpoint to which webhooks are sent
            api_key (str): the api key to send to the endpoint
            payload (Dict[str, Any]): the default payload that is sent to the endpoint
        """

        self.url = url
        self.api_key = api_key
        self.payload = payload

    def send(self, message: str):
        """Freely sends a message to the webhook endpoint

        Args:
            message (str): the message to send
        """

        data = {**self.payload, "content": message}

        # print("Sending webhook to " + self.url)

        params = json.dumps(data).encode("UTF-8")
        req = urllib.request.Request(
            self.url,
            data=params,
            headers={"content-type": "application/json", "authorization": self.api_key},
        )
        response = urllib.request.urlopen(req)

    def handle_event(self, event: Event):
        """Processes an event that happens during a batch run and converts it to a message.

        Args:
            event (Event): the event that occurred
        """

        event_type_translation = {0: "Batch run", 1: "Aggregation"}

        message = f"<b>{event.selected_sweep}</b>\n"

        if event.state == EventState.PROGRESS:
            message += (
                f"{event_type_translation[event.etype]} 🚶‍♂️‍➡️ {event.progress}%"
            )
        elif event.state == EventState.FINISHED:
            message += f"{event_type_translation[event.etype]} 🏁 FINISH !!"

        self.send(message)
