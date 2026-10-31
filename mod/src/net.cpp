#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include "net.hpp"

namespace {
    SOCKET s_sock = INVALID_SOCKET;
    SOCKET s_pending = INVALID_SOCKET;
    ULONGLONG s_lastAttempt = 0;
    bool s_wsaReady = false;

    void finishConnect(SOCKET sock) {
        u_long nonBlocking = 0;
        ioctlsocket(sock, FIONBIO, &nonBlocking);
        BOOL noDelay = TRUE;
        setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char*>(&noDelay), sizeof(noDelay));
        // The brain pauses the game while its network trains, so allow a long wait.
        DWORD recvTimeout = 120000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char*>(&recvTimeout), sizeof(recvTimeout));
        s_sock = sock;
    }
}

bool net::poll(uint16_t port) {
    if (s_sock != INVALID_SOCKET) return false;
    if (!s_wsaReady) {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
        s_wsaReady = true;
    }

    if (s_pending == INVALID_SOCKET) {
        ULONGLONG now = GetTickCount64();
        if (now - s_lastAttempt < 1000) return false;
        s_lastAttempt = now;

        s_pending = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (s_pending == INVALID_SOCKET) return false;
        u_long nonBlocking = 1;
        ioctlsocket(s_pending, FIONBIO, &nonBlocking);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        ::connect(s_pending, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    }

    // Check without waiting, so the game never stalls while brain.py is off.
    fd_set writeSet, errorSet;
    FD_ZERO(&writeSet);
    FD_ZERO(&errorSet);
    FD_SET(s_pending, &writeSet);
    FD_SET(s_pending, &errorSet);
    timeval noWait{0, 0};
    if (select(0, nullptr, &writeSet, &errorSet, &noWait) <= 0) {
        // Still connecting; give up after a second and retry later.
        if (GetTickCount64() - s_lastAttempt > 1000) {
            closesocket(s_pending);
            s_pending = INVALID_SOCKET;
        }
        return false;
    }

    int err = 0;
    int errLen = sizeof(err);
    getsockopt(s_pending, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&err), &errLen);
    SOCKET sock = s_pending;
    s_pending = INVALID_SOCKET;
    if (FD_ISSET(sock, &errorSet) || err != 0) {
        closesocket(sock);
        return false;
    }
    finishConnect(sock);
    return true;
}

bool net::connected() {
    return s_sock != INVALID_SOCKET;
}

void net::close() {
    if (s_sock != INVALID_SOCKET) {
        closesocket(s_sock);
        s_sock = INVALID_SOCKET;
    }
}

bool net::sendAll(void const* data, size_t len) {
    auto ptr = static_cast<char const*>(data);
    while (len > 0) {
        int sent = send(s_sock, ptr, static_cast<int>(len), 0);
        if (sent <= 0) {
            close();
            return false;
        }
        ptr += sent;
        len -= sent;
    }
    return true;
}

bool net::recvAll(void* data, size_t len) {
    auto ptr = static_cast<char*>(data);
    while (len > 0) {
        int got = recv(s_sock, ptr, static_cast<int>(len), 0);
        if (got <= 0) {
            close();
            return false;
        }
        ptr += got;
        len -= got;
    }
    return true;
}
