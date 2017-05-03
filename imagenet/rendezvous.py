import bisect
import random
import socket
import struct
import time
import torch
import torch.cuda.nccl2 as nccl2

MULTICAST_GROUP4 = ('224.66.41.62', 35552)
MULTICAST_GROUP6 = ('ff15:1e18:5d4c:4cf0:d02d:b659:53ba:b0a7', 35552)


def rendezvous(num_replicas, ttl=1):
    """Uses IP multicast to detect other replicas and exchange NCCL unique id

    Return: rank, device
    """

    uid = nccl2.get_unique_id()
    token = random.randint(0, 2**63 - 1)

    group = MULTICAST_GROUP6  # switch to MULTICAST_GROUP4 for IPv4
    addrinfo = socket.getaddrinfo(group[0], None)[0]

    tokens = set()
    uids = {}
    addresses = {}

    sendmsg = struct.pack('!q', token) + bytes(uid)

    with socket.socket(addrinfo[0], socket.SOCK_DGRAM) as sock:
        group_bin = socket.inet_pton(addrinfo[0], addrinfo[4][0])
        mreq = group_bin + struct.pack('I', 0)

        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', group[1]))
        sock.settimeout(10)

        if addrinfo[0] == socket.AF_INET:  # IPv4
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        else:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_MULTICAST_HOPS, ttl)
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_JOIN_GROUP, mreq)

        time.sleep(0.5)  # reduce chance that we send packet before joining group

        print('Finding other replicas...')
        sock.sendto(sendmsg, group)
        while len(tokens) != num_replicas:
            try:
                data, remote_addr = sock.recvfrom(256)
                remote_token = struct.unpack('!q', data[:8])[0]
                remote_uid = data[8:]
                if remote_token not in tokens:
                    tokens.add(remote_token)
                    uids[remote_token] = remote_uid
                    addresses[remote_token] = remote_addr
                    sock.sendto(sendmsg, group)
            except socket.timeout:
                sock.sendto(sendmsg, group)
                continue

    tokens = sorted(tokens)
    rank = bisect.bisect_left(tokens, token)
    if rank != 0:
        uid = uids[tokens[0]]

    def get_device():
        if torch.cuda.device_count() == 1:
            # If there is only one visible device, use it
            return 0
        dev = 0
        for t in tokens:
            if t == token:
                break
            if addresses[t][0] == addresses[token][0]:
                dev += 1
        return dev

    device = get_device()
    print('rank', rank, 'device', device)
    with torch.cuda.device(device):
        nccl2.initialize(num_replicas, uid, rank)
    return rank, device
