"""
radagent.federated
------------------
Federated learning implementation for RadAgent v2.

Implements FedAvg with SHA-256 audit chain for the Milan AI Week demo.
Author: Rayane Aggoune
"""
from radagent.federated.server import ClientUpdate, FederationRound, FedAvgServer
from radagent.federated.client import HospitalNode

__all__ = [
    "ClientUpdate",
    "FederationRound",
    "FedAvgServer",
    "HospitalNode",
]

# Made with Bob
