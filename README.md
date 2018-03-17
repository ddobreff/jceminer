# ethminer

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg)](https://github.com/RichardLitt/standard-readme)

> Ethereum miner with OpenCL, CUDA and stratum support. Cloned from ethrereum-mining/ethminer.

This is adapted for personal use. I make no claims of viability, and likely won't respond to bug reports or requests. Should
someone go to the trouble of pushing a decent change, I'll certainly take a look.

If you're going to try to use this, the release branches (1.0.0, 1.0.1, etc...) should be stable.

## Usage

For a full list of available command options, please run

```
ethminer --help

Options:
  -h [ --help ]                  produce help message.
  -d [ --devices ]               List devices.
  -v [ --version ]               list version.
  -f [ --file ] arg              Read parameters from file.
  -r [ --retries ] arg (=3)      Connection retries.
  -e [ --email ] arg             Stratum email.
  -w [ --timeout ] arg (=180)    Work timeout.
  --hash                         Report hashrate to pool.
  -s [ --stats-intvl ] arg (=15) statistics display interval.
  -l [ --stats-level ] arg (=0)  statistics display interval. 0 - HR only, 1 - 
                                 + fan & temp, 2 - + power.
  -p [ --pool ] arg              Pool URL.
                                 URL takes the form: scheme://[user[:password]@]hostname:port.
                                 unsecured schemes: stratum+tcp stratum1+tcp stratum2+tcp
                                 secured with any TLS: stratum+tls stratum1+tls stratum2+tls stratum+ssl stratum1+ssl stratum2+ssl
                                 secured with TLS 1.2: stratum+tls12 stratum1+tls12 stratum2+tls12
                                 Example: stratum+ssl://0x012345678901234567890234567890123.miner1@ethermine.org:5555
                                 Stratum versions:
                                 stratum  - official stratum spec: ethpool, ethermine, coinotron, mph, nanopool
                                 stratum1 - eth-proxy compatible: dwarfpool, f2pool, nanopool (required for hashrate reporting to work with nanopool)
                                 stratum2 - EthereumStratum/1.0.0: nicehash
  -a [ --api-port ] arg (=0)     API port number.
  --cl-plat arg (=0)             Opencl platform.
  --cl-devs arg                  Opencl device list.
  --cl-parallel arg (=8)         Opencl parallel hashes.
  --cl-kernel arg (=1)           Opencl kernel. 0 - Stable, 1 - Experimental, 2 - binary.
  --cl-tweak arg (=7)            Opencl wave tweak.
  --cl-global arg (=8192)        Opencl global work size. 0 - Auto.
  --cl-local arg (=64)           Opencl local work size.
  --cu-grid arg (=8192)          Cuda grid size.
  --cu-block arg (=128)          Cuda block size.
  --cu-devs arg                  Cuda device list.
  --cu-parallel arg (=4)         Cuda parallel hashes.
  --cu-sched arg (=4)            Cuda schedule mode. 0 - auto, 1 - spin, 2 - yield, 4 - sync
  --cu-stream arg (=2)           Cuda streams
  --cu-noeval                    Cuda bypass software result evaluation.
  --dag-mode arg (=0)            DAG load mode. 0 - parallel, 1 - sequential, 2 - single.
  --log-switch                   Log job switch time.
  --log-json                     Log formatted json messaging.
  -G [ --cl ]                    Opencl mode.
  -U [ --cu ]                    Cuda mode.
  -X [ --mixed ]                 Mixed opencl and cuda mode. Use OpenCL + CUDA in a system with mixed AMD/Nvidia cards. May require setting --cl-platform 1 or 2.
```

### Building from source

This project uses [CMake] and [Hunter] package manager.

1. Make sure git submodules are up to date

   ```sh
   git submodule update --init --recursive
   ```

2. Create a build directory.

   ```sh
   mkdir build; cd build
   ```

3. Configure the project with CMake. Check out additional
   [configuration options](#cmake-configuration-options).

   ```sh
   cmake ..
   ```

4. Build the project using [CMake Build Tool Mode]. This is a portable variant
   of `make`.

   ```sh
   cmake --build .
   ```

5. Optional, install the built executable.

   ```sh
   sudo make install
   ```

### CMake configuration options

Pass these options to CMake configuration command, e.g.

```sh
cmake .. -DETHASHCUDA=ON -DETHASHCL=OFF
```

- `-DETHASHCL=ON` - enable OpenCL mining, `ON` by default,
- `-DETHASHCUDA=ON` - enable CUDA mining, `OFF` by default.


## License

__Regrettably this must be licensed under the [GNU General Public License, Version 3](LICENSE).__

