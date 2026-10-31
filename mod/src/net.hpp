#pragma once
#include <cstddef>
#include <cstdint>

// Minimal blocking TCP client. Kept apart from Geode headers so winsock2.h
// doesn't clash with windows.h.
namespace net {
    // Call every frame while not connected. Starts a non-blocking connect
    // at most once per second and returns true on the frame it succeeds.
    bool poll(uint16_t port);
    bool connected();
    void close();
    bool sendAll(void const* data, size_t len);
    bool recvAll(void* data, size_t len);
}
