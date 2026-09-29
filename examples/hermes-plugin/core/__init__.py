"""Host-independent building blocks of the OpenViking Hermes plugin.

Modules in this package never import Hermes or the plugin package itself. Every
host-specific dependency (request headers, thread spawning, clocks) is passed in
by the caller, so each module can be loaded and tested on its own.
"""
