/*  Blah, blah, blah.. all this pedantic nonsense to say that this
    source code is made available under the terms and conditions
    of the accompanying GNU General Public License */

#include <chrono>
#include <thread>
#include <mongoose/mongoose.h>
#include <unistd.h>
#include <limits.h>
#include "restServer.h"
#include "libdevcore/Log.h"
#include "libdevcore/Common.h"
#include "miner-buildinfo.h"
#include <json/json.h>

using namespace dev;
using namespace eth;

restServer rest_server;

void restServer::reststats(stringstream& ss)
{
    Json::Value response;
    auto info = miner_get_buildinfo();
    char hostName[HOST_NAME_MAX + 1];
    gethostname(hostName, HOST_NAME_MAX + 1);
    string l = m_farm->farmLaunchedFormatted();
    response["version"] = info->project_version;
    response["hostname"] = hostName;
    WorkingProgress p = m_farm->miningProgress();
    response["gpus"] = p.minersHashes.size();
    using namespace std::chrono;
    double hashSum = 0.0;
    double powerSum = 0.0;
    for (unsigned i = 0; i < p.minersHashes.size(); i++) {
        double rate = p.minerRate(p.minersHashes[i]) / 1000000.0;
        hashSum += rate;
        if (i < p.minerMonitors.size()) {
            HwMonitor& hw(p.minerMonitors[i]);
            powerSum += hw.powerW;
        }
    }
    response["hashrate"] = round(hashSum);
    response["power"] = round(powerSum);
    stringstream sstats;
    sstats << m_farm->getSolutionStats();
    response["solutions"] = sstats.str();
    ss << response;
}

bool restServer::restgpu(stringstream& ss, unsigned index)
{
    Json::Value response;
    response["index"] = index;
    WorkingProgress p = m_farm->miningProgress();
    if (index >= p.minersHashes.size())
        return false;
    double rate = p.minerRate(p.minersHashes[index]) / 1000000.0;
    unsigned power = 0;
    unsigned fan = 0;
    if (index < p.minerMonitors.size()) {
        HwMonitor& hw(p.minerMonitors[index]);
        power = hw.powerW;
        fan = hw.fanP;
    }
    response["hashrate"] = round(rate * 10.0) / 10;
    response["power"] = power;
    response["fanpercent"] = fan;
    HwMonitorInfo& hw(m_farm->hwmoninfo(index));
    response["name"] = hw.deviceName;
    response["id"] = hw.deviceId;
    ss << response;
    return true;
}

static void ev_handler(struct mg_connection* c, int ev, void* p)
{

    if (ev == MG_EV_HTTP_REQUEST) {
        const char* gpu = "/gpu/";
        struct http_message* hm = (struct http_message*) p;
        stringstream content;
        if (mg_vcmp(&hm->uri, "/stats") == 0) {
            rest_server.reststats(content);
            mg_send_head(c, 200, content.str().length(), "Content-Type: application/json; charset=utf-8");
            mg_printf(c, "%s", content.str().c_str());
        }
        else if ((hm->uri.len > strlen(gpu)) && (memcmp(hm->uri.p, gpu, strlen(gpu)) == 0)) {
            using boost::lexical_cast;
            using boost::bad_lexical_cast;
            char num[8];
            unsigned l = hm->uri.len - strlen(gpu);
            if (l > sizeof(num) - 1)
                l = sizeof(num) - 1;
            memcpy(num, hm->uri.p + strlen(gpu), l);
            num[l] = 0;
            unsigned n;
            try {
                n = lexical_cast<unsigned>(num);
            }
            catch (bad_lexical_cast&) {
                mg_http_send_error(c, 404, nullptr);
                return;
            }
            if (rest_server.restgpu(content, n)) {
                mg_send_head(c, 200, content.str().length(), "Content-Type: application/json; charset=utf-8");
                mg_printf(c, "%s", content.str().c_str());
            }
            else
                mg_http_send_error(c, 404, nullptr);
        }
        else
            mg_http_send_error(c, 404, nullptr);
    }
}

void restServer::run(unsigned short port, dev::eth::Farm* farm, dev::eth::PoolManager* pool)
{
    m_farm = farm;
    m_pool = pool;
    m_port = to_string(port);
    new thread(bind(&restServer::run_thread, this));
}

void restServer::run_thread()
{
    struct mg_mgr mgr;
    struct mg_connection* c;

    mg_mgr_init(&mgr, NULL);
    loginfo("Starting restful server on port " << m_port);
    c = mg_bind(&mgr, m_port.c_str(), ev_handler);
    if (c == NULL) {
        logerror("Failed to create listener");
        return;
    }

    // Set up HTTP server parameters
    mg_set_protocol_http_websocket(c);

    for (;;)
        mg_mgr_poll(&mgr, 1000);
}

restServer::restServer()
{
}

restServer::~restServer()
{
}


