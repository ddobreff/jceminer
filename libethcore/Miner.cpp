/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#include "Miner.h"
#include "EthashAux.h"

using namespace dev;
using namespace eth;

unsigned dev::eth::Miner::s_dagLoadMode = 0;

unsigned dev::eth::Miner::s_dagLoadIndex = 0;

unsigned dev::eth::Miner::s_dagCreateDevice = 0;

uint8_t* dev::eth::Miner::s_dagInHostMemory = NULL;

bool g_logSwitchTime = false;
bool g_logJson = false;

std::ostream& operator<<(std::ostream& os, HwMonitor _hw)
{
    string power = "";
    if (_hw.powerW != 0) {
        ostringstream stream;
        stream << fixed << setprecision(0) << _hw.powerW << "W";
        power = stream.str();
    }
    os << _hw.tempC << "C " << _hw.fanP << "% " << power;
    return os;
}

