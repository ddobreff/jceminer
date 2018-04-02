/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#include "ApiServer.h"

#include <miner-buildinfo.h>
#include <libdevcore/Log.h>

ApiServer::ApiServer(AbstractServerConnector* conn, serverVersion_t type, Farm& farm,
                     bool& readonly) : AbstractServer(*conn, type), m_farm(farm)
{
    this->bindAndAddMethod(Procedure("miner_getstat1", PARAMS_BY_NAME, JSON_OBJECT, NULL), &ApiServer::getMinerStat1);
    this->bindAndAddMethod(Procedure("miner_getstathr", PARAMS_BY_NAME, JSON_OBJECT, NULL), &ApiServer::getMinerStatHR);
    if (!readonly) {
        this->bindAndAddMethod(Procedure("miner_restart", PARAMS_BY_NAME, JSON_OBJECT, NULL), &ApiServer::doMinerRestart);
        this->bindAndAddMethod(Procedure("miner_reboot", PARAMS_BY_NAME, JSON_OBJECT, NULL), &ApiServer::doMinerReboot);
    }
}

void ApiServer::getMinerStat1(const Json::Value& request, Json::Value& response)
{
    (void) request; // unused

    auto runningTime = std::chrono::duration_cast<std::chrono::minutes>(steady_clock::now() - this->m_farm.farmLaunched());

    SolutionStats s = m_farm.getSolutionStats();
    WorkingProgress p = m_farm.miningProgress();

    ostringstream totalMhEth;
    ostringstream totalMhDcr;
    ostringstream detailedMhEth;
    ostringstream detailedMhDcr;
    ostringstream tempAndFans;
    ostringstream poolAddresses;
    ostringstream invalidStats;

    totalMhEth << std::fixed << std::setprecision(0) << (p.rate() / 1000.0f) << ";" << s.getAccepts() << ";" <<
               s.getRejects();
    totalMhDcr << "0;0;0"; // DualMining not supported
    invalidStats << s.getFailures() << ";0"; // Invalid + Pool switches
    poolAddresses << m_farm.get_pool_addresses();
    invalidStats << ";0;0"; // DualMining not supported

    int gpuIndex = 0;
    int numGpus = p.minersHashes.size();
    for (auto const& i : p.minersHashes) {
        detailedMhEth << std::fixed << std::setprecision(0) << (p.minerRate(i) / 1000.0f) << (((
                          numGpus - 1) > gpuIndex) ? ";" : "");
        detailedMhDcr << "off" << (((numGpus - 1) > gpuIndex) ? ";" : ""); // DualMining not supported
        gpuIndex++;
    }

    int numMonGpus = p.minerMonitors.size();
    for (int gpuIndex = 0; gpuIndex < numGpus; gpuIndex++) {
        if (gpuIndex < numMonGpus) {
            auto mon = p.minerMonitors[gpuIndex];
            tempAndFans << mon.tempC << ";" << mon.fanP << ((gpuIndex < (numGpus - 1)) ? ";" : ""); // Fetching Temp and Fans
        }
        else
            tempAndFans << ((gpuIndex < (numGpus - 1)) ? "0;0;" : "0;0"); // Fetching Temp and Fans
    }

    response[0] = miner_get_buildinfo()->project_version;  //miner version.
    response[1] = toString(runningTime.count()); // running time, in minutes.
    response[2] =
        totalMhEth.str();              // total ETH hashrate in MH/s, number of ETH shares, number of ETH rejected shares.
    response[3] = detailedMhEth.str(); // detailed ETH hashrate for all GPUs.
    response[4] =
        totalMhDcr.str();              // total DCR hashrate in MH/s, number of DCR shares, number of DCR rejected shares.
    response[5] = detailedMhDcr.str(); // detailed DCR hashrate for all GPUs.
    response[6] = tempAndFans.str();   // Temperature and Fan speed(%) pairs for all GPUs.
    response[7] = poolAddresses.str(); // current mining pool. For dual mode, there will be two pools here.
    response[8] =
        invalidStats.str(); // number of ETH invalid shares, number of ETH pool switches, number of DCR invalid shares, number of DCR pool switches.
}

void ApiServer::getMinerStatHR(const Json::Value& request, Json::Value& response)
{
    (void) request; // unused

    //TODO:give key-value format
    auto runningTime = std::chrono::duration_cast<std::chrono::minutes>(steady_clock::now() - this->m_farm.farmLaunched());

    SolutionStats s = m_farm.getSolutionStats();
    WorkingProgress p = m_farm.miningProgress();

    ostringstream version;
    ostringstream runtime;
    Json::Value detailedMhEth;
    Json::Value detailedMhDcr;
    Json::Value temps;
    Json::Value fans;
    Json::Value powers;
    ostringstream poolAddresses;

    version << miner_get_buildinfo()->project_version;
    runtime << toString(runningTime.count());
    poolAddresses << m_farm.get_pool_addresses();

    int gpuIndex = 0;
    for (auto const& i : p.minersHashes) {
        detailedMhEth[gpuIndex] = (p.minerRate(i));
        gpuIndex++;
    }

    int numGpus = gpuIndex;
    int numMons = p.minerMonitors.size();
    for (gpuIndex = 0; gpuIndex < numGpus; gpuIndex++) {
        if (gpuIndex < numMons) {
            auto mon = p.minerMonitors[gpuIndex];
            temps[gpuIndex] = mon.tempC ; // Fetching Temps
            fans[gpuIndex] = mon.fanP; // Fetching Fans
            powers[gpuIndex] = int(mon.powerW); // Fetching Power
        }
        else {
            temps[gpuIndex] = 0; // Fetching Temps
            fans[gpuIndex] = 0; // Fetching Fans
            powers[gpuIndex] = 0; // Fetching Power
        }
    }

    response["version"] = version.str();        // miner version.
    response["runtime"] = runtime.str();        // running time, in minutes.
    // total ETH hashrate in MH/s, number of ETH shares, number of ETH rejected shares.
    response["ethhashrate"] = p.rate();
    response["ethhashrates"] = detailedMhEth;
    response["ethshares"]   = s.getAccepts();
    response["ethrejected"] = s.getRejects();
    response["ethinvalid"]  = s.getFailures();
    response["ethpoolsw"]   = 0;
    // Hardware Info
    response["temperatures"] = temps;                   // Temperatures(C) for all GPUs
    response["fanpercentages"] = fans;                  // Fans speed(%) for all GPUs
    response["powerusages"] = powers;                   // Power Usages(W) for all GPUs
    response["pooladdrs"] = poolAddresses.str();        // current mining pool. For dual mode, there will be two pools here.
}

void ApiServer::doMinerRestart(const Json::Value& request, Json::Value& response)
{
    (void) request; // unused
    (void) response; // unused
    logerror("Restart from API");
    exit(0);
}

void ApiServer::doMinerReboot(const Json::Value& request, Json::Value& response)
{
    (void) request; // unused
    (void) response; // unused

    // Not supported
}
