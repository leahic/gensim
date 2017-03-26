#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division  # py3 "true division"

import logging
import sys
import os
import heapq
from timeit import default_timer
from copy import deepcopy
from collections import namedtuple, defaultdict
import threading
import itertools

from gensim.utils import keep_vocab_item, call_on_class_only
from gensim.utils import date_transfer
from gensim.models.keyedvectors import KeyedVectors, Vocab

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray, vstack, logaddexp

from scipy.special import expit

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from six import iteritems, itervalues, string_types
from six.moves import xrange
from types import GeneratorType
from scipy import stats

logger = logging.getLogger(__name__)
FAST_VERSION = -1
MAX_WORDS_IN_BATCH = 10000

def train_document_dbow(model, doc_words, doctag_indexes, alpha, work=None,
                        train_words=False, learn_doctags=True, learn_words=True, learn_hidden=True,
                        word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
    if doctag_vectors is None:
        doctag_vectors = model.tagvecs.doctag_syn0
    if doctag_locks is None:
        doctag_locks = model.tagvecs.doctag_syn0_lockf

    if train_words and learn_words:
        train_batch_sg(model, [doc_words], alpha, work)
    for doctag_index in doctag_indexes:
        for word in doc_words:
            train_sg_pair(model, word, doctag_index, alpha, learn_vectors=learn_doctags,
                          learn_hidden=learn_hidden, context_vectors=doctag_vectors,
                          context_locks=doctag_locks)

    return len(doc_words)


def train_record_dbow():
    pass

def train_document_dm(model, doc_words, doctag_indexes, alpha, work=None, neu1=None,
                      learn_doctags=True, learn_words=True, learn_hidden=True,
                      word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
    """
    Update distributed memory model ("PV-DM") by training on a single document.

    Called internally from `Doc2Vec.train()` and `Doc2Vec.infer_vector()`. This
    method implements the DM model with a projection (input) layer that is
    either the sum or mean of the context vectors, depending on the model's
    `dm_mean` configuration field.  See `train_document_dm_concat()` for the DM
    model with a concatenated input layer.

    The document is provided as `doc_words`, a list of word tokens which are looked up
    in the model's vocab dictionary, and `doctag_indexes`, which provide indexes
    into the doctag_vectors array.

    Any of `learn_doctags', `learn_words`, and `learn_hidden` may be set False to
    prevent learning-updates to those respective model weights, as if using the
    (partially-)frozen model to infer other compatible vectors.

    This is the non-optimized, Python version. If you have a C compiler, gensim
    will use the optimized version from doc2vec_inner instead.

    """
    if word_vectors is None:
        word_vectors = model.wv.syn0
    if word_locks is None:
        word_locks = model.syn0_lockf
    if doctag_vectors is None:
        doctag_vectors = model.tagvecs.doctag_syn0
    if doctag_locks is None:
        doctag_locks = model.tagvecs.doctag_syn0_lockf

    word_vocabs = [model.wv.vocab[w] for w in doc_words if w in model.wv.vocab and
                   model.wv.vocab[w].sample_int > model.random.rand() * 2**32]

    for pos, word in enumerate(word_vocabs):
        reduced_window = model.random.randint(model.window)  # `b` in the original doc2vec code
        start = max(0, pos - model.window + reduced_window)
        window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
        word2_indexes = [word2.index for pos2, word2 in window_pos if pos2 != pos]
        l1 = np_sum(word_vectors[word2_indexes], axis=0) + np_sum(doctag_vectors[doctag_indexes], axis=0)
        count = len(word2_indexes) + len(doctag_indexes)
        if model.cbow_mean and count > 1 :
            l1 /= count
        neu1e = train_cbow_pair(model, word, word2_indexes, l1, alpha,
                                learn_vectors=False, learn_hidden=learn_hidden)
        if not model.cbow_mean and count > 1:
            neu1e /= count
        if learn_doctags:
            for i in doctag_indexes:
                doctag_vectors[i] += neu1e * doctag_locks[i]
        if learn_words:
            for i in word2_indexes:
                word_vectors[i] += neu1e * word_locks[i]

    return len(word_vocabs)

def train_record_dm():
    pass

class Itemtags(namedtuple('Itemtags' , 'itemid, date, tags')):
    __slots__ = ()

    def __str__(self):
        return str(itemid) + '\t' + str(date) + '\t' + ' '.join(map(str , tags))

class Itemindexes(namedtuple('Itemindexes' , 'itemid , tags')):
    __slots__ = ()

class TagvecsArray(utils.SaveLoad):
    def __init__(self, mapfile_path=None):
        self.doctags = {}  # string -> Doctag (only filled if necessary)
        self.max_rawint = -1  # highest rawint-indexed doctag
        self.offset2doctag = []  # int offset-past-(max_rawint+1) -> String (only filled if necessary)
        self.count = 0
        self.mapfile_path = mapfile_path

    def note_doctag(self, key, document_no, document_length):
        """Note a document tag during initial corpus scan, for structure sizing."""
        if isinstance(key, int):
            self.max_rawint = max(self.max_rawint, key)
        else:
            if key in self.doctags:
                self.doctags[key] = self.doctags[key].repeat(document_length)
            else:
                self.doctags[key] = Doctag(len(self.offset2doctag), document_length, 1)
                self.offset2doctag.append(key)
        self.count = self.max_rawint + 1 + len(self.offset2doctag)

    def indexed_doctags(self, doctag_tokens):
        """Return indexes and backing-arrays used in training examples."""
        return ([self._int_index(index) for index in doctag_tokens if index in self],
                self.doctag_syn0, self.doctag_syn0_lockf, doctag_tokens)

    def indexed_user(self , userid):
        return self._int_index('user_' + userid)

    def indexed_record(self , record):
        result = []
        for item in record:
            itemindex = self._int_index(item.itemid)
            tagindexes = [ self._int_index(tag) for tag in item.tags ]
            Itemindexes(itemindex , tagindexes)

    def trained_item(self, indexed_tuple):
        """Persist any changes made to the given indexes (matching tuple previously
        returned by indexed_doctags()); a no-op for this implementation"""
        pass

    def _int_index(self, index):
        """Return int index for either string or int index"""
        if isinstance(index, int):
            return index
        else:
            return self.max_rawint + 1 + self.doctags[index].offset

    def _key_index(self, i_index, missing=None):
        """Return string index for given int index, if available"""
        return self.index_to_doctag(i_index)

    def index_to_doctag(self, i_index):
        """Return string key for given i_index, if available. Otherwise return raw int doctag (same int)."""
        candidate_offset = i_index - self.max_rawint - 1
        if 0 <= candidate_offset < len(self.offset2doctag):
            return self.offset2doctag[candidate_offset]
        else:
            return i_index

    def __getitem__(self, index):
        """
        Accept a single key (int or string tag) or list of keys as input.

        If a single string or int, return designated tag's vector
        representation, as a 1D numpy array.

        If a list, return designated tags' vector representations as a
        2D numpy array: #tags x #vector_size.
        """
        if isinstance(index, string_types + (int,)):
            return self.doctag_syn0[self._int_index(index)]

        return vstack([self[i] for i in index])

    def __len__(self):
        return self.count

    def __contains__(self, index):
        if isinstance(index, int):
            return index < self.count
        else:
            return index in self.doctags

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm'])
        super(TagvecsArray, self).save(*args, **kwargs)

    def borrow_from(self, other_tagvecs):
        self.count = other_tagvecs.count
        self.doctags = other_tagvecs.doctags
        self.offset2doctag = other_tagvecs.offset2doctag

    def clear_sims(self):
        self.doctag_syn0norm = None

    def estimated_lookup_memory(self):
        """Estimated memory for tag lookup; 0 if using pure int tags."""
        return 60 * len(self.offset2doctag) + 140 * len(self.doctags)

    def reset_weights(self, model):
        length = max(len(self.doctags), self.count)
        if self.mapfile_path:
            self.doctag_syn0 = np_memmap(self.mapfile_path+'.doctag_syn0', dtype=REAL,
                                         mode='w+', shape=(length, model.vector_size))
            self.doctag_syn0_lockf = np_memmap(self.mapfile_path+'.doctag_syn0_lockf', dtype=REAL,
                                               mode='w+', shape=(length,))
            self.doctag_syn0_lockf.fill(1.0)
        else:
            self.doctag_syn0 = empty((length, model.vector_size), dtype=REAL)
            self.doctag_syn0_lockf = ones((length,), dtype=REAL)  # zeros suppress learning

        for i in xrange(length):
            # construct deterministic seed from index AND model seed
            seed = "%d %s" % (model.seed, self.index_to_doctag(i))
            self.doctag_syn0[i] = model.seeded_vector(seed)

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training or inference** after doing a replace.
        The model becomes effectively read-only = you can call `most_similar`, `similarity`
        etc., but not `train` or `infer_vector`.

        """
        if getattr(self, 'doctag_syn0norm', None) is None or replace:
            logger.info("precomputing L2-norms of doc weight vectors")
            if replace:
                for i in xrange(self.doctag_syn0.shape[0]):
                    self.doctag_syn0[i, :] /= sqrt((self.doctag_syn0[i, :] ** 2).sum(-1))
                self.doctag_syn0norm = self.doctag_syn0
            else:
                if self.mapfile_path:
                    self.doctag_syn0norm = np_memmap(
                        self.mapfile_path+'.doctag_syn0norm', dtype=REAL,
                        mode='w+', shape=self.doctag_syn0.shape)
                else:
                    self.doctag_syn0norm = empty(self.doctag_syn0.shape, dtype=REAL)
                np_divide(self.doctag_syn0, sqrt((self.doctag_syn0 ** 2).sum(-1))[..., newaxis], self.doctag_syn0norm)

    def most_similar(self, positive=[], negative=[], topn=10, clip_start=0, clip_end=None, indexer=None):
        self.init_sims()
        clip_end = clip_end or len(self.doctag_syn0norm)

        if isinstance(positive, string_types + integer_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each doc, if not already present; default to 1.0 for positive and -1.0 for negative docs
        positive = [
            (doc, 1.0) if isinstance(doc, string_types + (ndarray,) + integer_types)
            else doc for doc in positive
        ]
        negative = [
            (doc, -1.0) if isinstance(doc, string_types + (ndarray,) + integer_types)
            else doc for doc in negative
        ]

        # compute the weighted average of all docs
        all_docs, mean = set(), []
        for doc, weight in positive + negative:
            if isinstance(doc, ndarray):
                mean.append(weight * doc)
            elif doc in self.doctags or doc < self.count:
                mean.append(weight * self.doctag_syn0norm[self._int_index(doc)])
                all_docs.add(self._int_index(doc))
            else:
                raise KeyError("doc '%s' not in trained set" % doc)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        if indexer is not None:
            return indexer.most_similar(mean, topn)

        dists = dot(self.doctag_syn0norm[clip_start:clip_end], mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_docs), reverse=True)
        # ignore (don't return) docs from the input
        result = [(self.index_to_doctag(sim + clip_start), float(dists[sim])) for sim in best if (sim + clip_start) not in all_docs]
        return result[:topn]

    def doesnt_match(self, docs):
        """
        Which doc from the given list doesn't go with the others?

        (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        """
        self.init_sims()

        docs = [doc for doc in docs if doc in self.doctags or 0 <= doc < self.count]  # filter out unknowns
        logger.debug("using docs %s" % docs)
        if not docs:
            raise ValueError("cannot select a doc from an empty list")
        vectors = vstack(self.doctag_syn0norm[self._int_index(doc)] for doc in docs).astype(REAL)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
        dists = dot(vectors, mean)
        return sorted(zip(dists, docs))[0][1]

    def similarity(self, d1, d2):
        """
        Compute cosine similarity between two tagvecs in the trained set, specified by int index or
        string tag. (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        """
        return dot(matutils.unitvec(self[d1]), matutils.unitvec(self[d2]))

    def n_similarity(self, ds1, ds2):
        """
        Compute cosine similarity between two sets of tagvecs from the trained set, specified by int
        index or string tag. (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        """
        v1 = [self[doc] for doc in ds1]
        v2 = [self[doc] for doc in ds2]
        return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))

    def similarity_unseen_docs(self, model, doc_words1, doc_words2, alpha=0.1, min_alpha=0.0001, steps=5):
        """
        Compute cosine similarity between two post-bulk out of training documents.

        Document should be a list of (word) tokens.
        """
        d1 = model.infer_vector(doc_words=doc_words1, alpha=alpha, min_alpha=min_alpha, steps=steps)
        d2 = model.infer_vector(doc_words=doc_words2, alpha=alpha, min_alpha=min_alpha, steps=steps)
        return dot(matutils.unitvec(d1), matutils.unitvec(d2))


class Tag2Vec():

    def __init__(self, records=None, size=200, alpha=0.025, window=4,
                 max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 dm_mean=None,dm=1, dbow_words=0, dm_concat=0, dm_tag_count=1,
                 tagvecs=None, tagvecs_mapfile=None , comment=None, trim_rule=None,sorted_vocab=1,batch_words=MAX_WORDS_IN_BATCH, **kwargs):

        """
        from word2vec
        """

        sg=(1 + dm) % 2
        null_word=dm_concat

        self.load = call_on_class_only
        self.load_word2vec_format = call_on_class_only

        if FAST_VERSION == -1:
            logger.warning('Slow version of {0} is being used'.format(__name__))
        else:
            logger.debug('Fast version of {0} is being used'.format(__name__))

        self.initialize_word_vectors()
        self.sg = int(sg)
        self.cum_table = None  # for negative sampling
        self.vector_size = int(size)
        self.layer1_size = int(size)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.min_alpha_yet_reached = float(alpha)  # To warn user if alpha increases
        self.window = int(window)
        self.max_vocab_size = max_vocab_size
        self.seed = seed
        self.random = random.RandomState(seed)
        self.sample = sample
        self.workers = int(workers)
        self.min_alpha = float(min_alpha)
        self.hs = 1 - sg % 2
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.train_count = 0
        self.total_train_time = 0
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.model_trimmed_post_training = False

        """
        from doc2vec
        """

        if dm_mean is not None:
            self.cbow_mean = dm_mean

        self.dbow_words = dbow_words
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count
        if self.dm and self.dm_concat:
            self.layer1_size = (self.dm_tag_count + (2 * self.window)) * self.vector_size

        self.tagvecs = tagvecs or TagvecsArray(tagvecs_mapfile)
        self.comment = comment
        self.user_list , self.document = self.rebuild(records)
        if self.document is not None:
            self.build_vocab(self.document, trim_rule=trim_rule)
            self.train(self.document)

    @property
    def dm(self):
        return not self.sg  # opposite of SG

    @property
    def dbow(self):
        return self.sg  # same as SG

    def rebuild(self , records):
        record_no = -1
        total_words = 0
        min_reduce = 1
        user_list = set()
        document = dict()
        checked_string_types = 0
        for record_no, record in enumerate(records):
            if not checked_string_types:
                if not isinstance(record, string_types):
                    logger.warn("Each 'record' should be a string[%s]", type(record))
                checked_string_types += 1
            try:
                userid , tags , itemid , date_str = record.strip().split('\t')
            except:
                logger.warn("PROGRESS: at record #%i , parse failed" , record_no)
                continue
            date_value = date_transfer(date_str)
            if userid not in user_list:
                user_list.add(userid)
                document[userid] = []
            document[userid].append( Itemtags( itemid , date_value , tags.split()) )

        for userid in document:
            document[userid] = sorted(document[userid] , key = lambda x : x.date_value)

        return user_list , document

    def initialize_word_vectors(self):
        self.wv = KeyedVectors()

    def make_cum_table(self, power=0.75, domain=2**31 - 1):
        """
        Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.

        To draw a word index, choose a random integer up to the maximum value in the
        table (cum_table[-1]), then finding that integer's sorted insertion point
        (as if by bisect_left or ndarray.searchsorted()). That insertion point is the
        drawn index, coming up in proportion equal to the increment at that slot.

        Called internally from 'build_vocab()'.
        """
        vocab_size = len(self.wv.index2word)
        self.cum_table = zeros(vocab_size, dtype=uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0
        for word_index in xrange(vocab_size):
            train_words_pow += self.wv.vocab[self.wv.index2word[word_index]].count**power
        cumulative = 0.0
        for word_index in xrange(vocab_size):
            cumulative += self.wv.vocab[self.wv.index2word[word_index]].count**power
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

    def create_binary_tree(self):
        """
        Create a binary Huffman tree using stored vocabulary word counts. Frequent words
        will have shorter binary codes. Called internally from `build_vocab()`.

        """
        logger.info("constructing a huffman tree from %i words", len(self.wv.vocab))

        # build the huffman tree
        heap = list(itervalues(self.wv.vocab))
        heapq.heapify(heap)
        for i in xrange(len(self.wv.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + len(self.wv.vocab), left=min1, right=min2))

        # recurse over the tree, assigning a binary code to each vocabulary word
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < len(self.wv.vocab):
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = array(list(points) + [node.index - len(self.wv.vocab)], dtype=uint32)
                    stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
                    stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))

            logger.info("built huffman tree with maximum node depth %i", max_depth)


    def build_vocab(self, document, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        self.scan_vocab(document, progress_per=progress_per, trim_rule=trim_rule)  # initial survey
        self.scale_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, update=update)  # trim by min_count & precalculate downsampling
        self.finalize_vocab(update=update)  # build tables & arrays

    def scan_vocab(self, document, progress_per=10000, trim_rule=None):
        vocab = defaultdict(int)
        for userid in document:
            vocab['user_' + userid] += 1
            for record in document[userid]:
                vocab['item_' + record.itemid] += 1
                for tag in record.tags:
                    vocab['tag_' + tag] += 1

        self.corpus_count = len(document)
        self.raw_vocab = vocab

    def scale_vocab(self, min_count=None, sample=None, dry_run=False, keep_raw_vocab=False, trim_rule=None, update=False):

        self.wv.index2word = []
        # make stored settings match these applied settings
        self.min_count = min_count
        self.sample = sample
        self.wv.vocab = {}

        for word, v in iteritems(self.raw_vocab):
                self.wv.vocab[word] = Vocab(count=v, index=len(self.wv.index2word))
                self.wv.index2word.append(word)

    def finalize_vocab(self, update=False):
        """Build tables and model weights based on final vocabulary settings."""
        if not self.wv.index2word:
            self.scale_vocab()
        if self.sorted_vocab and not update:
            self.sort_vocab()
        if self.hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()
        if self.null_word:
            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
            # this word is only ever input – never predicted – so count, huffman-point, etc doesn't matter
            word, v = '\0', Vocab(count=1, sample_int=0)
            v.index = len(self.wv.vocab)
            self.wv.index2word.append(word)
            self.wv.vocab[word] = v
        # set initial input/projection and hidden weights
        if not update:
            self.reset_weights()
        else:
            self.update_weights()


    def clear_sims(self):
        self.tagvecs.clear_sims()

    def sort_vocab(self):
        """Sort the vocabulary so the most frequent words have the lowest indexes."""
        if len(self.wv.syn0):
            raise RuntimeError("must sort before initializing vectors/weights")
        self.wv.index2word.sort(key=lambda word: self.wv.vocab[word].count, reverse=True)
        for i, word in enumerate(self.wv.index2word):
            self.wv.vocab[word].index = i

    def reset_weights(self):
        """
        from word2vec
        """
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        self.wv.syn0 = empty((len(self.wv.vocab), self.vector_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in xrange(len(self.wv.vocab)):
            # construct deterministic seed from word AND seed argument
            self.wv.syn0[i] = self.seeded_vector(self.wv.index2word[i] + str(self.seed))
        if self.hs:
            self.syn1 = zeros((len(self.wv.vocab), self.layer1_size), dtype=REAL)
        if self.negative:
            self.syn1neg = zeros((len(self.wv.vocab), self.layer1_size), dtype=REAL)
        self.wv.syn0norm = None

        self.syn0_lockf = ones(len(self.wv.vocab), dtype=REAL)  # zeros suppress learning

        """
        from doc2vec
        """

        if self.dm and self.dm_concat:
            # expand l1 size to match concatenated tags+words length
            self.layer1_size = (self.dm_tag_count + (2 * self.window)) * self.vector_size
            logger.info("using concatenative %d-dimensional layer1" % (self.layer1_size))

        self.tagvecs.reset_weights(self)

    def seeded_vector(self, seed_string):
        """Create one 'random' vector (but deterministic by seed_string)"""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = random.RandomState(self.hashfxn(seed_string) & 0xffffffff)
        return (once.rand(self.vector_size) - 0.5) / self.vector_size

    def reset_from(self, other_model):
        """
        from doc2vec
        """
        """Reuse shareable structures from other_model."""
        self.tagvecs.borrow_from(other_model.tagvecs)

        """
        from word2vec
        """
        self.wv.vocab = other_model.vocab
        self.wv.index2word = other_model.index2word
        self.cum_table = other_model.cum_table
        self.corpus_count = other_model.corpus_count
        self.reset_weights()

    def _do_train_job(self, job_batch, alpha, inits):
        work, neu1 = inits
        tally = 0
        for userid , record in job:
            indexed_doctags = self.tagvecs.indexed_doctags(doc.tags)
            doctag_indexes, doctag_vectors, doctag_locks, ignored = indexed_doctags
            if self.sg:
                tally += train_record_dbow(self, userid , record , alpha, work,
                                             train_words=self.dbow_words,
                                             doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            else:
                tally += train_record_dm(self, userid , record, alpha, work, neu1,
                                           doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
        return tally, self._raw_word_count(job)

    def _record_word_count(self, record):
        num = 0
        if isinstance(record , list):
            for item in record:
                num += 1 + len(item.tags)
        return num

    def _document_word_count(self, document):
        num = 0
        if isinstace(record , dict):
            for userid in document:
                num += 1
                for item in document[userid]:
                    num += 1 + len(item.tags)
        return num

    def infer_vector(self, doc_words, alpha=0.1, min_alpha=0.0001, steps=5):
        """
        Infer a vector for given post-bulk training document.

        Document should be a list of (word) tokens.
        """
        doctag_vectors = empty((1, self.vector_size), dtype=REAL)
        doctag_vectors[0] = self.seeded_vector(' '.join(doc_words))
        doctag_locks = ones(1, dtype=REAL)
        doctag_indexes = [0]

        work = zeros(self.layer1_size, dtype=REAL)
        if not self.sg:
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)

        for i in range(steps):
            if self.sg:
                train_document_dbow(self, doc_words, doctag_indexes, alpha, work,
                                    learn_words=False, learn_hidden=False,
                                    doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            elif self.dm_concat:
                train_document_dm_concat(self, doc_words, doctag_indexes, alpha, work, neu1,
                                         learn_words=False, learn_hidden=False,
                                         doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            else:
                train_document_dm(self, doc_words, doctag_indexes, alpha, work, neu1,
                                  learn_words=False, learn_hidden=False,
                                  doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            alpha = ((alpha - min_alpha) / (steps - i)) + min_alpha

        return doctag_vectors[0]

    def estimate_memory(self, vocab_size=None, report=None):

        """
        from doc2vec
        """
        report = report or {}
        report['doctag_lookup'] = self.tagvecs.estimated_lookup_memory()
        report['doctag_syn0'] = self.tagvecs.count * self.vector_size * dtype(REAL).itemsize


        """
        from word2vec
        """
        vocab_size = vocab_size or len(self.wv.vocab)
        report = report or {}
        report['vocab'] = vocab_size * (700 if self.hs else 500)
        report['syn0'] = vocab_size * self.vector_size * dtype(REAL).itemsize
        if self.hs:
            report['syn1'] = vocab_size * self.layer1_size * dtype(REAL).itemsize
        if self.negative:
            report['syn1neg'] = vocab_size * self.layer1_size * dtype(REAL).itemsize
        report['total'] = sum(report.values())
        logger.info("estimated required memory for %i words and %i dimensions: %i bytes",
                    vocab_size, self.vector_size, report['total'])


        return report


    def train(self, document, total_words=None, word_count=0,
              total_examples=None, queue_factor=2, report_delay=1.0):
        """
        Update the model's neural weights from a sequence of document (can be a once-only generator stream).
        For Word2Vec, each sentence must be a list of unicode strings. (Subclasses may accept other examples.)

        To support linear learning-rate decay from (initial) alpha to min_alpha, either total_examples
        (count of document) or total_words (count of raw words in document) should be provided, unless the
        document are the same as those that were used to initially build the vocabulary.

        """
        if (self.model_trimmed_post_training):
            raise RuntimeError("Parameters for training were discarded using model_trimmed_post_training method")
        if FAST_VERSION < 0:
            import warnings
            self.neg_labels = []
            if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.

        logger.info(
            "training model with %i workers on %i vocabulary and %i features, "
            "using sg=%s hs=%s sample=%s negative=%s window=%s",
            self.workers, len(self.wv.vocab), self.layer1_size, self.sg,
            self.hs, self.sample, self.negative, self.window)

        if not self.wv.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")
        if not len(self.wv.syn0):
            raise RuntimeError("you must first finalize vocabulary before training the model")

        if not hasattr(self, 'corpus_count'):
            raise ValueError(
                "The number of document in the training corpus is missing. Did you load the model via load_word2vec_format?"
                "Models loaded via load_word2vec_format don't support further training. "
                "Instead start with a blank model, scan_vocab on the new corpus, intersect_word2vec_format with the old model, then train.")

        if total_words is None and total_examples is None:
            if self.corpus_count:
                total_examples = self.corpus_count
            else:
                raise ValueError("you must provide either total_words or total_examples, to enable alpha and progress calculations")

        job_tally = 0

        if self.iter > 1:
            #multi_document = utils.RepeatCorpusNTimes(document, self.iter)
            total_words = total_words and total_words * self.iter
            total_examples = total_examples and total_examples * self.iter

        def worker_loop():
            """Train the model, lifting lists of document from the job_queue."""
            work = matutils.zeros_aligned(self.layer1_size, dtype=REAL)  # per-thread private work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            jobs_processed = 0
            while True:
                job = job_queue.get()
                if job is None:
                    progress_queue.put(None)
                    break  # no more jobs => quit this worker
                job_batch, alpha = job
                tally, raw_tally = self._do_train_job(job_batch , alpha, (work, neu1))
                progress_queue.put((self.raw_word_count(record), tally, raw_tally))  # report back progress
                jobs_processed += 1
            logger.debug("worker exiting, processed %i jobs", jobs_processed)

        def job_producer():
            """Fill jobs queue using the input `document` iterator."""
            job_batch, batch_size = [], 0
            pushed_words, pushed_examples = 0, 0
            next_alpha = self.alpha
            if next_alpha > self.min_alpha_yet_reached:
                logger.warn("Effective 'alpha' higher than previous training cycles")
            self.min_alpha_yet_reached = next_alpha
            job_no = 0
            for _ in range(self.iter):
                for user_idx, userid in enumerate(document):
                    record_length = self._record_word_count(document[userid]) + 1

                    # can we fit this sentence into the existing job batch?
                    if batch_size + record_length <= self.batch_words:
                        # yes => add it to the current job
                        job_batch.append( (userid , document[userid]) )
                        batch_size += record_length
                    else:
                        # no => submit the existing job
                        logger.debug(
                            "queueing job #%i (%i words, %i document) at alpha %.05f",
                            job_no, batch_size, len(job_batch), next_alpha)
                        job_no += 1
                        job_queue.put((job_batch, next_alpha))

                        # update the learning rate for the next job
                        if self.min_alpha < next_alpha:
                            if total_examples:
                                # examples-based decay
                                pushed_examples += len(job_batch)
                                progress = 1.0 * pushed_examples / total_examples
                            else:
                                # words-based decay
                                pushed_words += sum([1 + self._record_word_count(temp2) for temp1 , temp2 in job_batch])
                                progress = 1.0 * pushed_words / total_words
                            next_alpha = self.alpha - (self.alpha - self.min_alpha) * progress
                            next_alpha = max(self.min_alpha, next_alpha)

                        # add the sentence that didn't fit as the first item of a new job
                        job_batch, batch_size = [(userid , document[userid])], record_length

            # add the last job too (may be significantly smaller than batch_words)
            if job_batch:
                logger.debug(
                    "queueing job #%i (%i words, %i document) at alpha %.05f",
                    job_no, batch_size, len(job_batch), next_alpha)
                job_no += 1
                job_queue.put((job_batch, next_alpha))

            if job_no == 0 and self.train_count == 0:
                logger.warning(
                    "train() called with an empty iterator (if not intended, "
                    "be sure to provide a corpus that offers restartable "
                    "iteration = an iterable)."
                )

            # give the workers heads up that they can finish -- no more work!
            for _ in xrange(self.workers):
                job_queue.put(None)
            logger.debug("job loop exiting, total %i jobs", job_no)

        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        unfinished_worker_count = len(workers)
        workers.append(threading.Thread(target=job_producer))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        example_count, trained_word_count, raw_word_count = 0, 0, word_count
        start, next_report = default_timer() - 0.00001, 1.0

        while unfinished_worker_count > 0:
            report = progress_queue.get()  # blocks if workers too slow
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
                logger.info("worker thread finished; awaiting finish of %i more threads", unfinished_worker_count)
                continue
            examples, trained_words, raw_words = report
            job_tally += 1

            # update progress stats
            example_count += examples
            trained_word_count += trained_words  # only words in vocab & sampled
            raw_word_count += raw_words

            # log progress once every report_delay seconds
            elapsed = default_timer() - start
            if elapsed >= next_report:
                if total_examples:
                    # examples-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * example_count / total_examples, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue))
                else:
                    # words-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * raw_word_count / total_words, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue))
                next_report = elapsed + report_delay

        # all done; report the final stats
        elapsed = default_timer() - start
        logger.info(
            "training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed)
        if job_tally < 10 * self.workers:
            logger.warn("under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay")

        # check that the input corpus hasn't changed during iteration
        if total_examples and total_examples != example_count:
            logger.warn("supplied example count (%i) did not equal expected count (%i)", example_count, total_examples)
        if total_words and total_words != raw_word_count:
            logger.warn("supplied raw word count (%i) did not equal expected count (%i)", raw_word_count, total_words)

        self.train_count += 1  # number of times train() has been called
        self.total_train_time += elapsed
        self.clear_sims()
        return trained_word_count

    def __str__(self):
        """Abbreviated name reflecting major configuration paramaters."""
        segments = []
        if self.comment:
            segments.append('"%s"' % self.comment)
        if self.sg:
            if self.dbow_words:
                segments.append('dbow+w')  # also training words
            else:
                segments.append('dbow')  # PV-DBOW (skip-gram-style)

        else:  # PV-DM...
            if self.dm_concat:
                segments.append('dm/c')  # ...with concatenative context layer
            else:
                if self.cbow_mean:
                    segments.append('dm/m')
                else:
                    segments.append('dm/s')
        segments.append('d%d' % self.vector_size)  # dimensions
        if self.negative:
            segments.append('n%d' % self.negative)  # negative samples
        if self.hs:
            segments.append('hs')
        if not self.sg or (self.sg and self.dbow_words):
            segments.append('w%d' % self.window)  # window size, when relevant
        if self.min_count > 1:
            segments.append('mc%d' % self.min_count)
        if self.sample > 0:
            segments.append('s%g' % self.sample)
        if self.workers > 1:
            segments.append('t%d' % self.workers)
        return '%s(%s)' % (self.__class__.__name__, ','.join(segments))

    def delete_temporary_training_data(self, keep_doctags_vectors=True, keep_inference=True):
        if not keep_inference:
            self._minimize_model(False, False, False)
        if self.tagvecs and hasattr(self.tagvecs, 'doctag_syn0') and not keep_doctags_vectors:
            del self.tagvecs.doctag_syn0
        if self.tagvecs and hasattr(self.tagvecs, 'doctag_syn0_lockf'):
            del self.tagvecs.doctag_syn0_lockf
