#include <chrono>
#include <mongoose/mongoose.h>
#include "httpServer.h"
#include "libdevcore/Log.h"

using namespace dev;
using namespace eth;

httpServer::httpServer(unsigned short port, Farm& farm) : m_port(port), m_farm(farm)
{
	if (m_port)
		new std::thread(httpServer::serve, this);
}

static const char* s_http_port = "40001";
static struct mg_serve_http_opts s_http_server_opts;

httpServer::~httpServer()
{
}

static void getstat1(stringstream& ss)
{
	ss << "Hi!";
}

static void ev_handler(struct mg_connection* c, int ev, void* p)
{

	if (ev == MG_EV_HTTP_REQUEST) {
		struct http_message* hm = (struct http_message*) p;
		char uri[32];
		unsigned uriLen = hm->uri.len;
		if (uriLen >= sizeof(uri) - 1)
			uriLen = sizeof(uri) - 1;
		memcpy(uri, hm->uri.p, uriLen);
		uri[uriLen] = 0;
		if (::strcmp(uri, "/getstat1"))
			mg_http_send_error(c, 404, nullptr);
		else {
			stringstream content;
			getstat1(content);
			mg_send_head(c, 200, (int)content.str().length(), "Content-Type: text/plain");
			mg_printf(c, "%.*s", (int)content.str().length(), content.str().c_str());
		}
	}
}
void httpServer::serve(httpServer* server)
{
	(void)server;
	struct mg_mgr mgr;
	struct mg_connection* nc;

	stringstream ss;
	ss << server->m_port;

	s_http_port = strdup(ss.str().c_str());

	mg_mgr_init(&mgr, NULL);
	loginfo << "Starting web server on port " << s_http_port << '\n';
	nc = mg_bind(&mgr, s_http_port, ev_handler);
	if (nc == NULL) {
		logerror << "Failed to create listener\n";
		return;
	}

	// Set up HTTP server parameters
	mg_set_protocol_http_websocket(nc);
	s_http_server_opts.document_root = ".";  // Serve current directory
	s_http_server_opts.enable_directory_listing = "no";

	for (;;)
		mg_mgr_poll(&mgr, 1000);

}
