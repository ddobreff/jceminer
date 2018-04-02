/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#pragma once

#include <thread>
#include <libethcore/Farm.h>
#include <libproto/PoolManager.h>

class httpServer
{
public:
    httpServer();
    ~httpServer();
    void run(unsigned short port, dev::eth::Farm* farm, dev::eth::PoolManager* pool);
    void run_thread();
    void getstat1(stringstream& ss);

    dev::eth::Farm* m_farm;
    dev::eth::PoolManager* m_pool;
    std::string m_port;
private:
    void tableHeader(stringstream& ss, unsigned columns);
};

extern httpServer http_server;

