#pragma once

#include <thread>
#include <libethcore/Farm.h>

class httpServer
{
public:
	httpServer();
	~httpServer();
	void run(unsigned short port, dev::eth::Farm* farm);
	void run_thread();
	void getstat1(stringstream& ss);

	dev::eth::Farm* m_farm;
	std::string m_port;
};

extern httpServer http_server;

