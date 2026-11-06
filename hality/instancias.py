from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Instancia:
    nome: str
    vcpus: int
    ram_gb: int


CATALOGO: dict[str, Instancia] = {i.nome: i for i in (
    Instancia("t3.medium", 2, 4),
    Instancia("t3.large", 2, 8),
    Instancia("t3.xlarge", 4, 16),
    Instancia("c7i.large", 2, 4),
    Instancia("c7i.xlarge", 4, 8),
    Instancia("c7i.2xlarge", 8, 16),
    Instancia("c7i.4xlarge", 16, 32),
    Instancia("m7i.large", 2, 8),
    Instancia("m7i.xlarge", 4, 16),
    Instancia("g4dn.xlarge", 4, 16),
)}

NENHUMA = "nenhuma"


def buscar(nome: str) -> Instancia:
    try:
        return CATALOGO[nome]
    except KeyError:
        raise ValueError(f"Instancia desconhecida: {nome}. Validas: {', '.join(CATALOGO)}") from None


def threads_para(inst: Instancia) -> int:
    return max(1, min(inst.vcpus, os.cpu_count() or inst.vcpus))


def rss_bytes() -> int:
    with open("/proc/self/status") as f:
        for linha in f:
            if linha.startswith("VmRSS:"):
                return int(linha.split()[1]) * 1024
    return 0


def ram_total_bytes() -> int:
    with open("/proc/meminfo") as f:
        for linha in f:
            if linha.startswith("MemTotal:"):
                return int(linha.split()[1]) * 1024
    return 0


def demo() -> None:
    assert buscar("c7i.xlarge").vcpus == 4
    try:
        buscar("x1.inexistente")
    except ValueError as e:
        assert "Validas" in str(e)
    else:
        raise AssertionError("aceitou instancia inexistente")
    assert 1 <= threads_para(buscar("c7i.4xlarge")) <= (os.cpu_count() or 16)
    assert rss_bytes() > 0
    assert ram_total_bytes() > rss_bytes()
    print("ok - catalogo de instancias")


if __name__ == "__main__":
    demo()
