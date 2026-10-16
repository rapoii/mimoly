"""Fail-closed network guard shared by the offline script runners.

Install before importing production modules. HTTPX ASGITransport remains usable;
real socket/DNS and HTTPX network transports are forbidden, even if application
code catches Exception. No credentials are needed for these tests.
"""
import socket
from contextvars import ContextVar
from unittest.mock import patch

import httpx


class NetworkForbidden(BaseException):
    pass


ATTEMPTS = []
_PATCHES = []
_SOCKETPAIR = ContextVar('offline_socketpair', default=False)
_ORIGINAL_SOCKETPAIR = socket.socketpair
_ORIGINAL_CONNECT = socket.socket.connect


def local_socketpair(*args, **kwargs):
    # Windows asyncio creates its self-pipe via the stdlib loopback fallback.
    # Allow only that synchronous socketpair operation, never arbitrary loopback.
    token = _SOCKETPAIR.set(True)
    try:
        return _ORIGINAL_SOCKETPAIR(*args, **kwargs)
    finally:
        _SOCKETPAIR.reset(token)


def guarded_connect(sock, address):
    if _SOCKETPAIR.get():
        return _ORIGINAL_CONNECT(sock, address)
    return deny_network()


def deny_network(*args, **kwargs):
    ATTEMPTS.append('network transport')
    raise NetworkForbidden('Offline test attempted network access')


async def deny_async_network(*args, **kwargs):
    deny_network()


def install_network_deny():
    if _PATCHES:
        return
    for target, replacement in (
        ('socket.socketpair', local_socketpair),
        ('socket.socket.connect', guarded_connect),
        ('socket.socket.connect_ex', deny_network),
        ('socket.create_connection', deny_network),
        ('socket.getaddrinfo', deny_network),
        ('httpx.HTTPTransport.handle_request', deny_network),
        ('httpx.AsyncHTTPTransport.handle_async_request', deny_async_network),
    ):
        guard = patch(target, replacement)
        guard.start()
        _PATCHES.append(guard)
