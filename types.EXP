(*
    BSD Socket API for IEC 61131-3 Structured Text

    Copyright (c) X-Pie Software GmbH
    Written by VB.
*)

TYPE
    SOCK_DOMAIN:    (AF_UNSPEC, AF_INET) := AF_UNSPEC;
    SOCK_TYPE:      (SOCK_STREAM);
    PROTOCOL:       (TCP);

    SOCKADDR: STRUCT
        addr:       DINT := -1;
        port:       UINT := 0;
    END_STRUCT;

    CONNECT_STATE:  (
                        NOT_CONNECTED,
                        BOUND,
                        LISTENING,
                        CONNECTED
                    ) := NOT_CONNECTED;

    PAL_STATE:      (
                        PAL_NULL,
                        PAL_BIND,
                        PAL_LISTEN,
                        PAL_ACCEPT,
                        PAL_CONNECT,
                        PAL_RECV,
                        PAL_SEND,
                        PAL_READY,
                        PAL_CLOSE
                    ) := PAL_NULL;

    SOCK_FD: STRUCT
        domain:     SOCK_DOMAIN;
        stype:      SOCK_TYPE;
        proto:      PROTOCOL;
        s_addr:     SOCKADDR;
        d_addr:     SOCKADDR;
        state:      CONNECT_STATE;
    END_STRUCT;

    POLLFD: STRUCT
        fd:         INT;
        events:     INT;
        revents:    INT;
    END_STRUCT;

    NFDS_T:         INT;

    POLL_EVENT:     (
                        POLLERR,
                        POLLHUP,
                        POLLIN,
                        POLLNVAL,
                        POLLOUT
                    );

    ERR_TYPE: (
        EOK,
        EINTR,
        EMFILE,
        EBADF,
        EAFNOSUPPORT,
        ECONNREFUSED,
        EHOSTUNREACH,
        EOPNOTSUPP,
        EINVAL,
        EISCONN,
        EDESTADDRREQ
    ) := EINTR; (* by default everything is asynch *)

    IFACE_PAL_API: STRUCT
        fd:         INT;
        addr:       SOCKADDR;
        backlog:    INT;
        buffer:     UDINT;
        length:     UDINT;
        flags:      INT;
        state:      PAL_STATE;
    END_STRUCT;
END_TYPE

