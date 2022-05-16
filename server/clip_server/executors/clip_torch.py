import os
import warnings
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Optional, Dict

import numpy as np
import torch
from clip_server.executors.helper import (
    split_img_txt_da,
    preproc_image,
    preproc_text,
    set_rank,
)
from clip_server.model import clip
from jina import Executor, requests, DocumentArray


class CLIPEncoder(Executor):
    def __init__(
        self,
        name: str = 'ViT-B/32',
        device: Optional[str] = None,
        jit: bool = False,
        num_worker_preprocess: int = 4,
        minibatch_size: int = 16,
        **kwargs,
    ):
        """Encode images and texts into embeddings where the input is an iterable of raw strings.

        Each image and text must be represented as a string. The following strings are acceptable:

            - local image filepath, will be considered as an image
            - remote image http/https, will be considered as an image
            - a dataURI, will be considered as an image
            - plain text, will be considered as a sentence

        :param name: Model weights, default is ViT-B/32. Support all OpenAI released pretrained models
        :param device: cuda or cpu. Default is None means auto-detect.
        :param jit: If to enable Torchscript JIT, default is False.
        :param num_worker_preprocess: The number of CPU workers for image & text prerpocessing, default 4.
        :param minibatch_size: The size of a minibatch for CPU preprocessing and GPU encoding, default 64. Reduce the size of it if you encounter OOM on GPU.
        :param kwargs: kwargs passed to super().
        :return: the embedding in a numpy ndarray with shape ``[N, D]``. ``N`` is in the same length of ``content``.
        """
        super().__init__(**kwargs)

        if not device:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device

        if not self._device.startswith('cuda') and (
            'OMP_NUM_THREADS' not in os.environ
            and hasattr(self.runtime_args, 'replicas')
        ):
            replicas = getattr(self.runtime_args, 'replicas', 1)
            num_threads = max(1, torch.get_num_threads() // replicas)
            if num_threads < 2:
                warnings.warn(
                    f'Too many replicas ({replicas}) vs too few threads {num_threads} may result in '
                    f'sub-optimal performance.'
                )

            # NOTE: make sure to set the threads right after the torch import,
            # and `torch.set_num_threads` always take precedence over environment variables `OMP_NUM_THREADS`.
            # For more details, please see https://pytorch.org/docs/stable/generated/torch.set_num_threads.html
            # FIXME: This hack would harm the performance in K8S deployment.
            torch.set_num_threads(max(num_threads, 1))
            torch.set_num_interop_threads(1)

        self._minibatch_size = minibatch_size
        self._model, self._preprocess_tensor = clip.load(
            name, device=self._device, jit=jit
        )

        self._pool = ThreadPool(processes=num_worker_preprocess)

    @requests(on='/rank')
    async def rank(self, docs: 'DocumentArray', parameters: Dict, **kwargs):
        """Computes a relevance score for each match using cosine scores and softmax.

        :param docs: List of :class:`DocumentArray` matched to query results.
        :param parameters: Not used (kept to maintain interface).
        :param kwargs: Not used (kept to maintain interface).
        """
        await self.encode(docs['@r,m'])

        set_rank(docs)

    @requests
    async def encode(self, docs: 'DocumentArray', **kwargs):
        """Encode images and texts into embeddings where the input is an iterable of :class:`docarray.Document`.

        :param docs: An iterable of :class:`docarray.Document`, each Document must be filled with `.uri`, `.text` or `.blob`.
        :param kwargs: Not used (kept to maintain interface).
        """
        _img_da = DocumentArray()
        _txt_da = DocumentArray()
        for d in docs:
            split_img_txt_da(d, _img_da, _txt_da)

        with torch.inference_mode():
            # for image
            if _img_da:
                for minibatch in _img_da.map_batch(
                    partial(
                        preproc_image,
                        preprocess_fn=self._preprocess_tensor,
                        device=self._device,
                        return_np=False,
                    ),
                    batch_size=self._minibatch_size,
                    pool=self._pool,
                ):
                    minibatch.embeddings = (
                        self._model.encode_image(minibatch.tensors)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

            # for text
            if _txt_da:
                for minibatch, _texts in _txt_da.map_batch(
                    partial(preproc_text, device=self._device, return_np=False),
                    batch_size=self._minibatch_size,
                    pool=self._pool,
                ):
                    minibatch.embeddings = (
                        self._model.encode_text(minibatch.tensors)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    minibatch.texts = _texts

        # drop tensors
        docs.tensors = None
        return docs
