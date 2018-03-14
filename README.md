# ethminer

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg)](https://github.com/RichardLitt/standard-readme)

> Ethereum miner with OpenCL, CUDA and stratum support. Cloned from ethrereum-mining/ethminer.

This is adapted for personal use. I make no claims of viability, and likely won't respond to bug reports or requests. Should
someone go to the trouble of pushing a decent change, I'll certainly take a look.

## Usage

For a full list of available command options, please run

```
ethminer --help


Usage ethminer [OPTIONS]
Options:

 Mining configuration:
    -G,--cl  When mining use the GPU via OpenCL.
    -U,--cu  When mining use the GPU via CUDA.
    -X,--cu-cl Use OpenCL + CUDA in a system with mixed AMD/Nvidia cards. May require setting --cl-platform 1 or 2.
        Use --list-devices option to check which platform is your AMD.
    -l, --list-devices List the detected OpenCL/CUDA devices and exit. Should be combined with -G or -U flag
    --stats-interval <n> Set mining stats display interval in seconds. (default: every 5 seconds)
    -L, --dag-load-mode <mode> DAG generation mode.
        parallel    - load DAG on all GPUs at the same time (default)
        sequential  - load DAG on GPUs one after another. Use this when the miner crashes during DAG generation
        single <n>  - generate DAG on device n, then copy to other devices
    --work-timeout <n> reconnect/failover after n seconds of working on the same job. Defaults to 180. Don't set lower
        than max. avg. block time
    -RH, --report-hashrate Report current hashrate to pool (please only enable on pools supporting this)
    -HWMON [<n>], Displays gpu temp, fan percent and power usage. Note: In linux, the program uses sysfs, which may
        require running with root priviledges.
        0: Displays only temp and fan percent (default)
        1: Also displays power usage
    --exit Stops the miner whenever an error is encountered
    -SE, --stratum-email <s> Email address used in eth-proxy (optional)
    -P URL Specify a pool URL. Can be used multiple times. The 1st for for the primary pool, and the 2nd for the failover pool.
        URL takes the form: scheme://[user[:password]@]hostname:port.
          unsecured schemes:    stratum+tcp stratum1+tcp stratum2+tcp
          secured with any TLS: stratum+tls stratum1+tls stratum2+tls
          secured with TLS 1.2: stratum+ssl stratum+tls12 stratum1+ssl stratum1+tls12 stratum2+ssl stratum2+tls12
        Example: stratum+ssl://0x012345678901234567890234567890123.miner1@ethermine.org:5555
        Stratum versions:
          stratum  - official stratum spec: ethpool, ethermine, coinotron, mph, nanopool
          stratum1 - eth-proxy compatible: dwarfpool, f2pool, nanopool (required for hashrate reporting to work with nanopool)
          stratum2 - EthereumStratum/1.0.0: nicehash

 OpenCL configuration:
    --cl-platform <n>  When mining using -G/--cl use OpenCL platform n (default: 0).
    --cl-device <n>  When mining using -G/--cl use OpenCL device n (default: 0).
    --cl-devices <0 1 ..n> Select which OpenCL devices to mine on. Default is to use all
    --cl-kernel <n>  Use a different OpenCL kernel (default: use stable kernel)
        0: stable kernel
        1: experimental kernel
        2: binary kernel
    --cl-local-work Set the OpenCL local work size. Default is 128
    --cl-global-work Set the OpenCL global work size as a multiple of the local work size. Default is 8192 * 128
        You may also specify auto for optimal Radeon value based on configuration.
    --cl-parallel-hash <1 2 ..8> Define how many threads to associate per hash. Default=8
    --cl-wavetweak 0-100 

 CUDA configuration:
    --cu-block-size Set the CUDA block work size. Default is 128
    --cu-grid-size Set the CUDA grid size. Default is 8192
    --cu-streams Set the number of CUDA streams. Default is 2
    --cu-schedule <mode> Set the schedule mode for CUDA threads waiting for CUDA devices to finish work.
         Default is 'sync'. Possible values are:
        auto  - Uses a heuristic based on the number of active CUDA contexts in the process C and the number of logical
            processors in the system P. If C > P, then yield else spin.
        spin  - Instruct CUDA to actively spin when waiting for results from the device.
        yield - Instruct CUDA to yield its thread when waiting for results from the device.
        sync  - Instruct CUDA to block the CPU thread on a synchronization primitive when waiting for the results
            from the device.
    --cu-devices <0 1 ..n> Select which CUDA GPUs to mine on. Default is to use all
    --cu-parallel-hash <1 2 ..8> Define how many hashes to calculate in a kernel, can be scaled to achieve better
            performance. Default=4
    --cu-noeval  bypass host software re-evalution of GPU solutions.
        This will trim some milliseconds off the time it takes to send a result to the pool.
        Use at your own risk! If GPU generates errored results they WILL be forwarded to the pool
        Not recommended at high overclock.

 API core configuration:
    --api-port Set the api port, the miner should listen to. Use 0 to disable. Default=0, use negative numbers to
         run in readonly mode. for example -3333.

 Logging:
    --log-switch Display per card switch time.
    --log-json Display formatted json I/O.

 General Options:

    -V,--version  Show the version and exit.
    -h,--help  Show this help message and exit.

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

