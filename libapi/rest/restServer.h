/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#pragma once

#include <thread>
#include <libethcore/Farm.h>
#include <libproto/PoolManager.h>

class restServer
{
public:
    restServer();
    ~restServer();
    void run(unsigned short port, dev::eth::Farm* farm, dev::eth::PoolManager* pool);
    void run_thread();
    void reststats(stringstream& ss);
    bool restgpu(stringstream& ss, unsigned index);

    dev::eth::Farm* m_farm;
    dev::eth::PoolManager* m_pool;
    std::string m_port;
};

extern restServer rest_server;

