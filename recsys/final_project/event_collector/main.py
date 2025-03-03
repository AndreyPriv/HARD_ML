import asyncio
import json
import time

import aio_pika
from aio_pika import Message
from aio_pika.abc import AbstractRobustExchange, AbstractRobustConnection
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import InteractEvent
from watched_filter import WatchedFilter
import uvicorn

app = FastAPI()
watched_filter = WatchedFilter()

queue_name = "user_interactions"
routing_key = "user.interact.message"
exchange = "user.interact"

_rabbitmq_connection: AbstractRobustConnection = None
_rabbitmq_exchange = None

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthcheck")
def read_root():
    return True


@app.post("/interact")
async def interact(message: InteractEvent):
    message.timestamp = time.time()
    await publish_message(
        Message(
            bytes(json.dumps(message.model_dump()), "utf-8"),
            content_type="text/json",
        )
    )

    return 200


async def create_rabbitmq_exchange() -> AbstractRobustExchange:
    global _rabbitmq_exchange, _rabbitmq_connection
    if _rabbitmq_exchange is None or _rabbitmq_connection.is_closed:
        _rabbitmq_connection = await aio_pika.connect_robust(
            "amqp://guest:guest@rabbitmq/", loop=asyncio.get_event_loop()
        )

        channel = await _rabbitmq_connection.channel()

        _rabbitmq_exchange = await channel.declare_exchange(
            "user.interact", type="direct"
        )

        queue = await channel.declare_queue(queue_name)

        await queue.bind(_rabbitmq_exchange, routing_key)
    return _rabbitmq_exchange


async def publish_message(message: Message):
    rabbitmq_exchange = await create_rabbitmq_exchange()
    await rabbitmq_exchange.publish(
        message,
        routing_key,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)