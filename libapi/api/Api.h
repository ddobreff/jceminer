/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#pragma once

#include "ApiServer.h"
#include <libethcore/Farm.h>
#include <libethcore/Miner.h>
#include <jsonrpccpp/server/connectors/tcpsocketserver.h>

using namespace jsonrpc;
using namespace dev;
using namespace dev::eth;

class Api
{
public:
    Api(const int& port, Farm& farm);
private:
    ApiServer* m_server;
    Farm& m_farm;
};

