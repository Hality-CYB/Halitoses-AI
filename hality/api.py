from __future__ import annotations

import io
import os
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from .instancias import NENHUMA, Instancia, buscar, ram_total_bytes, rss_bytes, threads_para
from .pipeline import Hality

TAMANHO_MAX = 20 * 1024 * 1024
AVISO = "Triagem, nao diagnostico. Procure um dentista para avaliacao."
SIMULACAO_ATIVA = os.environ.get("HALITY_SIMULACAO") == "1"
FOLGA_RAM = 0.85

_modelo: Hality | None = None
_trava_simulacao = threading.Lock()

REQUISICOES = Counter("hality_requisicoes_total", "Analises concluidas",
                      ["veredito", "classe", "instancia"])
LATENCIA = Histogram("hality_latencia_segundos", "Tempo de analise", ["instancia"],
                     buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 60))
INSTANCIA_ATUAL = Gauge("hality_instancia_simulada", "Instancia EC2 simulada na ultima analise",
                        ["instancia"])
VCPUS = Gauge("hality_vcpus_instancia", "vCPUs da instancia simulada")
RAM_INSTANCIA = Gauge("hality_ram_instancia_bytes", "RAM da instancia simulada")
THREADS = Gauge("hality_threads_torch", "Threads do torch na ultima analise")


def _publicar_instancia(nome: str, vcpus: int, ram_bytes: int) -> None:
    INSTANCIA_ATUAL.clear()
    INSTANCIA_ATUAL.labels(nome).set(1)
    VCPUS.set(vcpus)
    RAM_INSTANCIA.set(ram_bytes)


def _aquecer(m: Hality) -> None:
    buf = io.BytesIO()
    Image.fromarray(np.full((256, 256, 3), 128, np.uint8)).save(buf, "PNG")
    m.analisar(buf.getvalue())
    if m.gate is not None:
        m.tongue_gate(np.full((256, 256, 3), 128, np.uint8))


def _executar(raw: bytes, inst: Instancia | None):
    if inst is None:
        THREADS.set(torch.get_num_threads())
        return _modelo.analisar(raw)
    with _trava_simulacao:
        anterior = torch.get_num_threads()
        torch.set_num_threads(threads_para(inst))
        THREADS.set(torch.get_num_threads())
        try:
            return _modelo.analisar(raw)
        finally:
            torch.set_num_threads(anterior)


@asynccontextmanager
async def ciclo(app: FastAPI):
    global _modelo
    _modelo = Hality()
    await run_in_threadpool(_aquecer, _modelo)
    _publicar_instancia(NENHUMA, os.cpu_count() or 0, ram_total_bytes())
    yield


app = FastAPI(
    title="Hality",
    description="Triagem de indicio de halitose a partir de foto da lingua. Nao e diagnostico.",
    version="0.3.0",
    lifespan=ciclo,
)
app.mount("/metrics", make_asgi_app())


@app.get("/saude")
def saude() -> dict:
    m = _modelo
    if m is None:
        raise HTTPException(503, "Modelos ainda carregando.")
    return {
        "status": "ok",
        "segmentador": m.seg.nome,
        "gate": m.gate is not None,
        "auc_teste": round(m.auc_teste, 3),
        "iou_segmentador": round(m.iou_seg, 3),
        "limiar": round(m.limiar, 3),
        "faixa_abstencao": [round(m.abst_lo, 3), round(m.abst_hi, 3)],
        "simulacao_ativa": SIMULACAO_ATIVA,
        "aviso": AVISO,
    }


@app.post("/analisar")
async def analisar(
    foto: UploadFile = File(..., description="Foto da lingua"),
    instancia: str | None = Form(None, description="Teste: nome da EC2 a simular, ex c7i.xlarge"),
) -> JSONResponse:
    inst = None
    if instancia:
        if not SIMULACAO_ATIVA:
            raise HTTPException(403, "Simulacao desligada. Suba a API com HALITY_SIMULACAO=1.")
        try:
            inst = buscar(instancia)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    raw = await foto.read()
    if not raw:
        raise HTTPException(400, "Arquivo vazio.")
    if len(raw) > TAMANHO_MAX:
        raise HTTPException(413, f"Arquivo acima de {TAMANHO_MAX // 1024 // 1024} MB.")
    if _modelo is None:
        raise HTTPException(503, "Modelos ainda carregando.")

    nome = inst.nome if inst else NENHUMA
    if inst:
        _publicar_instancia(inst.nome, inst.vcpus, inst.ram_gb * 1024 ** 3)

    t0 = time.perf_counter()
    r = await run_in_threadpool(_executar, raw, inst)
    dt = time.perf_counter() - t0

    LATENCIA.labels(nome).observe(dt)
    REQUISICOES.labels(r.veredito, str(r.classe or "-"), nome).inc()

    corpo = r.dict()
    corpo["arquivo"] = foto.filename
    corpo["tempo_ms"] = round(dt * 1000)
    corpo["aviso"] = AVISO
    if inst:
        rss = rss_bytes()
        corpo["simulacao"] = {
            "instancia": inst.nome,
            "vcpus": inst.vcpus,
            "threads_usadas": threads_para(inst),
            "ram_gb": inst.ram_gb,
            "ram_processo_mb": round(rss / 1024 ** 2),
            "cabe_na_ram": rss < inst.ram_gb * 1024 ** 3 * FOLGA_RAM,
        }
    return JSONResponse(corpo, status_code=200)
