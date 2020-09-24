#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy
import asyncio
import websockets
from vosk import Model, KaldiRecognizer
from tools import WorkerStreaming

# create WorkerStreaming object
worker = WorkerStreaming()

# Load ASR models (acoustic model and decoding graph)
worker.log.info('Load acoustic model and decoding graph')
model = Model(worker.AM_PATH, worker.LM_PATH, worker.CONFIG_FILES_PATH+"/online.conf")

# decode chunk audio
def process_chunk(rec, message):
    if message == '{"eof" : 1}':
        return rec.FinalResult(), True
    elif rec.AcceptWaveform(message):
        return rec.Result(), False
    else:
        return rec.PartialResult(), False


# Recognizer
async def recognize(websocket, path):
    rec = None
    audio = b''
    nbrSpk = 10
    sample_rate = model.GetSampleFrequecy() # get default sample frequency
    metadata = worker.METADATA
    while True:
        try:
            data = await websocket.recv()
            
            # Load configuration if provided
            if isinstance(data, str) and 'config' in data:
                jobj = json.loads(data)['config']
                if 'sample_rate' in jobj:
                    sample_rate = float(jobj['sample_rate'])
                if 'metadata' in jobj:
                    metadata = bool(jobj['metadata'])
                if 'nbrSpeakers' in jobj:
                    nbrSpk = bool(jobj['nbrSpeakers'])
                continue

            # Create the recognizer, word list is temporary disabled since not every model supports it
            if not rec:
                rec = KaldiRecognizer(model, sample_rate, metadata)

            if not isinstance(data, str):
                audio = audio + data

            response, stop = process_chunk(rec, data)
            await websocket.send(response)
            if stop:
                if metadata:
                    obj = rec.GetMetadata()
                    data = json.loads(obj)
                    response = worker.process_metadata(data, nbrSpk)
                    await websocket.send(response)
                break
        except Exception as e:
            worker.log.error(e)
            break

if __name__ == '__main__':
    worker.log.info("Server is listening on port "+str(worker.SERVICE_PORT))
    start_server = websockets.serve(recognize, "0.0.0.0", worker.SERVICE_PORT)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()