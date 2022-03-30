"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np
import time


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma, agent_num):
    s_ = lnet.pad_pack_rnn(s_)
    if done:
        v_s_ = np.zeros([1, 1])  # terminal
    else:
        v_s_ = lnet.forward(s_)[-1].data.numpy()

    buffer_v_target = []
    for r in br:  # reverse buffer r
        # print(f'r shape {r}  v_s_.shape: { v_s_.shape}')
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    s, a, v_t = np.vstack(bs), \
                v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)), \
                v_wrap(np.array(buffer_v_target)[:, None].reshape(-1, 1))
    tt1 = time.time()

    # print(f's.shape {s.shape} a.shape {a.shape}  v_t.shape{v_t.shape}')

    loss = lnet.loss_func(s, a, v_t)
    # print(f'loss: {loss}')
    tt0 = time.time()
    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    tt1 = time.time()
    print(f'stage1: {tt1 - tt0}')
    loss.backward()
    tt2 = time.time()
    print(f'stage2: {tt2 - tt1}')
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    tt3 = time.time()
    print(f'stage3: {tt3 - tt2}')
    opt.step()
    tt4 = time.time()
    print(f'stage4: {tt4 - tt3}')

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r,
           res_queue, name, writer, ratio, flights_num, time_t, cancelled_flights):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01

    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )

    writer.add_scalars(f'UAM'
                       f'(agent{flights_num}_{time_t.tm_mon}{time_t.tm_mday}{time_t.tm_hour}{time_t.tm_min}',
                       {'global_ep_r_value': global_ep_r.value,
                        'resolved_overdemand_ratio': ratio,
                        'cancelled_flights:': cancelled_flights},
                       global_ep.value)
