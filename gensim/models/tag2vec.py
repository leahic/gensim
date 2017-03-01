#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
import os
import warnings

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from collections import namedtuple, defaultdict
from timeit import default_timer

from numpy import zeros, random, sum as np_sum, add as np_add, concatenate, \
    repeat as np_repeat, array, float32 as REAL, empty, ones, memmap as np_memmap, \
    sqrt, newaxis, ndarray, dot, vstack, dtype, divide as np_divide


from gensim.utils import call_on_class_only
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.models.word2vec import Word2Vec, train_cbow_pair, train_sg_pair, train_batch_sg
from six.moves import xrange, zip
from six import string_types, integer_types, itervalues

logger = logging.getLogger(__name__)

try:
    from gensim.models.doc2vec_inner import train_document_dbow, train_document_dm, train_document_dm_concat
    from gensim.models.word2vec_inner import FAST_VERSION  # blas-adaptation shared from word2vec
    logger.debug('Fast version of {0} is being used'.format(__name__))
except ImportError:
    logger.warning('Slow version of {0} is being used'.format(__name__))
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1

    def train_document_dbow(model, doc_words, doctag_indexes, alpha, work=None,
                            train_words=False, learn_doctags=True, learn_words=True, learn_hidden=True,
                            word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf

        if train_words and learn_words:
            train_batch_sg(model, [doc_words], alpha, work)
        for doctag_index in doctag_indexes:
            for word in doc_words:
                train_sg_pair(model, word, doctag_index, alpha, learn_vectors=learn_doctags,
                              learn_hidden=learn_hidden, context_vectors=doctag_vectors,
                              context_locks=doctag_locks)

        return len(doc_words)

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
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf

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

    def train_document_dm_concat(model, doc_words, doctag_indexes, alpha, work=None, neu1=None,
                                 learn_doctags=True, learn_words=True, learn_hidden=True,
                                 word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """
        Update distributed memory model ("PV-DM") by training on a single document, using a
        concatenation of the context window word vectors (rather than a sum or average).

        Called internally from `Doc2Vec.train()` and `Doc2Vec.infer_vector()`.

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
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf

        word_vocabs = [model.wv.vocab[w] for w in doc_words if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2**32]
        doctag_len = len(doctag_indexes)
        if doctag_len != model.dm_tag_count:
            return 0  # skip doc without expected number of doctag(s) (TODO: warn/pad?)

        null_word = model.wv.vocab['\0']
        pre_pad_count = model.window
        post_pad_count = model.window
        padded_document_indexes = (
            (pre_pad_count * [null_word.index])  # pre-padding
            + [word.index for word in word_vocabs if word is not None]  # elide out-of-Vocabulary words
            + (post_pad_count * [null_word.index])  # post-padding
        )

        for pos in range(pre_pad_count, len(padded_document_indexes) - post_pad_count):
            word_context_indexes = (
                padded_document_indexes[(pos - pre_pad_count): pos]  # preceding words
                + padded_document_indexes[(pos + 1):(pos + 1 + post_pad_count)]  # following words
            )
            word_context_len = len(word_context_indexes)
            predict_word = model.wv.vocab[model.wv.index2word[padded_document_indexes[pos]]]
            # numpy advanced-indexing copies; concatenate, flatten to 1d
            l1 = concatenate((doctag_vectors[doctag_indexes], word_vectors[word_context_indexes])).ravel()
            neu1e = train_cbow_pair(model, predict_word, None, l1, alpha,
                                    learn_hidden=learn_hidden, learn_vectors=False)

            # filter by locks and shape for addition to source vectors
            e_locks = concatenate((doctag_locks[doctag_indexes], word_locks[word_context_indexes]))
            neu1e_r = (neu1e.reshape(-1, model.vector_size)
                       * np_repeat(e_locks, model.vector_size).reshape(-1, model.vector_size))

            if learn_doctags:
                np_add.at(doctag_vectors, doctag_indexes, neu1e_r[:doctag_len])
            if learn_words:
                np_add.at(word_vectors, word_context_indexes, neu1e_r[doctag_len:])

        return len(padded_document_indexes) - pre_pad_count - post_pad_count


class Itemtags(namedtuple('Itemtags' , 'item, date, taglist')):
    __slots__ = ()

    def __str__(self):
        return str(item) + '\t' + str(date) + '\t' + ' '.join(map(str , taglist))

class TagvecsArray(utils.SaveLoad):
    def __init__(self, mapfile_path=None):
        self.doctags = {}  # string -> Tagtag (only filled if necessary)
        self.max_rawint = -1  # highest rawint-indexed doctag
        self.offset2doctag = []  # int offset-past-(max_rawint+1) -> String (only filled if necessary)
        self.count = 0
        self.mapfile_path = mapfile_path

    def note_doctag(self, key, document_no, document_length):
        if isinstance(key, int):
            self.max_rawint = max(self.max_rawint, key)
        else:
            if key in self.doctags:
                self.doctags[key] = self.doctags[key].repeat(document_length)
            else:
                self.doctags[key] = Tagtag(len(self.offset2doctag), document_length, 1)
                self.offset2doctag.append(key)
        self.count = self.max_rawint + 1 + len(self.offset2doctag)

    def indexed_doctags(self, doctag_tokens):
        return ([self._int_index(index) for index in doctag_tokens if index in self],
                self.doctag_syn0, self.doctag_syn0_lockf, doctag_tokens)

    def trained_item(self, indexed_tuple):
        pass

    def _int_index(self, index):
        if isinstance(index, int):
            return index
        else:
            return self.max_rawint + 1 + self.doctags[index].offset

    def _key_index(self, i_index, missing=None):
        warnings.warn("use TagvecsArray.index_to_doctag", DeprecationWarning)
        return self.index_to_doctag(i_index)

    def index_to_doctag(self, i_index):
        candidate_offset = i_index - self.max_rawint - 1
        if 0 <= candidate_offset < len(self.offset2doctag):
            return self.offset2doctag[candidate_offset]
        else:
            return i_index

    def __getitem__(self, index):
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
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'table', 'cum_table'])

        super(Tag2vec, self).save(*args, **kwargs)

    def borrow_from(self, other_docvecs):
        self.count = other_docvecs.count
        self.doctags = other_docvecs.doctags
        self.offset2doctag = other_docvecs.offset2doctag

    def clear_sims(self):
        self.doctag_syn0norm = None

    def estimated_lookup_memory(self):
        """Estimated memory for tag lookup; 0 if using pure int tags."""
        return 60 * len(self.offset2doctag) + 140 * len(self.doctags)

    def reset_weights(self, model):
        """
        from word2vec
        """
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

        self.syn0_lockf = ones(len(self.wv.vocab), dtype=REAL)
        """
        from doc2vec
        """
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
        """
        Find the top-N most similar docvecs known from training. Positive docs contribute
        positively towards the similarity, negative docs negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given docs. Docs may be specified as vectors, integer indexes
        of trained docvecs, or if the documents were originally presented with string tags,
        by the corresponding tags.

        The 'clip_start' and 'clip_end' allow limiting results to a particular contiguous
        range of the underlying doctag_syn0norm vectors. (This may be useful if the ordering
        there was chosen to be significant, such as more popular tag IDs in lower indexes.)
        """
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
        Compute cosine similarity between two docvecs in the trained set, specified by int index or
        string tag. (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        """
        return dot(matutils.unitvec(self[d1]), matutils.unitvec(self[d2]))

    def n_similarity(self, ds1, ds2):
        """
        Compute cosine similarity between two sets of docvecs from the trained set, specified by int
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
    """Class for training, using and evaluating neural networks described in http://arxiv.org/pdf/1405.4053v2.pdf"""
    def __init__(self, records=None, dm_mean=None,
                 dm=1, dbow_words=0, dm_concat=0, dm_tag_count=1,
                 uservecs=None, uservecs_mapfile=None,
                 tagvecs=None, tagvecs_mapfile=None,
                 itemvecs=None, itemvecs_mapfile=None, comment=None, trim_rule=None, **kwargs):

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
        self.min_count = min_count
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

        self.uservecs = uservecs or TagvecsArray(uservecs_mapfile)
        self.tagvecs = tagvecs or TagvecsArray(tagvecs_mapfile)
        self.itemvecs = itemvecs or TagvecsArray(itemvecs_mapfile)
        self.comment = comment
        self.user_list , self.documents = self.rebuild(records)
        if self.documents is not None:
            self.build_vocab(self.documents, trim_rule=trim_rule)
            self.train(self.user_list , self.documents)

    @property
    def dm(self):
        return not self.sg  # opposite of SG

    @property
    def dbow(self):
        def rebuild(self , records):
        return self.sg  # same as SG

        record_no = -1
        total_words = 0
        min_reduce = 1
        user_list = set()
        document = dict()
        checked_string_types = 0
        for record_no, record in enumerate(records):
            if not checked_string_types:
                if isinstance(record, string_types):
                    logger.warn("Each 'record' item should be a list of words (usually unicode strings)."
                                "First item here is instead plain %s.", type(sentence))
                checked_string_types += 1
            if record_no % progress_per == 0:
                logger.info("PROGRESS: at record #%i, processed %i words, keeping %i word types",
                            record_no, sum(itervalues(vocab)) + total_words, len(vocab))
            try:
                userid , tags , itemid , date_str = record.strip().split('\t')
            except:
                logger.warn("PROGRESS: at record #%i , parse failed" , record_no)
                continue
            date_value = util.date_transfer(date_str)
            if userid in user_list:
            else:
                user_list.insert(user_id)
                document[user_id] = []
            document[user_id].append( Itemtags( itemid , date_value , tags.split()) )

        return user_list , document




    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        self.scan_vocab(sentences, progress_per=progress_per, trim_rule=trim_rule)  # initial survey
        self.scale_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, update=update)  # trim by min_count & precalculate downsampling
        self.finalize_vocab(update=update)  # build tables & arrays

    def scan_vocab(self, sentences, progress_per=10000, trim_rule=None):
        """Do an initial scan of all words appearing in sentences."""
        logger.info("collecting all words and their counts")
        sentence_no = -1
        total_words = 0
        min_reduce = 1
        vocab = defaultdict(int)
        checked_string_types = 0
        for sentence_no, sentence in enumerate(sentences):
            if not checked_string_types:
                if isinstance(sentence, string_types):
                    logger.warn("Each 'sentences' item should be a list of words (usually unicode strings)."
                                "First item here is instead plain %s.", type(sentence))
                checked_string_types += 1
            if sentence_no % progress_per == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words, keeping %i word types",
                            sentence_no, sum(itervalues(vocab)) + total_words, len(vocab))
            for word in sentence:
                vocab[word] += 1

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                total_words += utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1

        total_words += sum(itervalues(vocab))
        logger.info("collected %i word types from a corpus of %i raw words and %i sentences",
                    len(vocab), total_words, sentence_no + 1)
        self.corpus_count = sentence_no + 1
        self.raw_vocab = vocab

    def scale_vocab(self, min_count=None, sample=None, dry_run=False, keep_raw_vocab=False, trim_rule=None, update=False):
        """
        Apply vocabulary settings for `min_count` (discarding less-frequent words)
        and `sample` (controlling the downsampling of more-frequent words).

        Calling with `dry_run=True` will only simulate the provided settings and
        report the size of the retained vocabulary, effective corpus length, and
        estimated memory requirements. Results are both printed via logging and
        returned as a dict.

        Delete the raw vocabulary after the scaling is done to free up RAM,
        unless `keep_raw_vocab` is set.

        """
        min_count = min_count or self.min_count
        sample = sample or self.sample
        drop_total = drop_unique = 0

        if not update:
            logger.info("Loading a fresh vocabulary")
            retain_total, retain_words = 0, []
            # Discard words less-frequent than min_count
            if not dry_run:
                self.wv.index2word = []
                # make stored settings match these applied settings
                self.min_count = min_count
                self.sample = sample
                self.wv.vocab = {}

            for word, v in iteritems(self.raw_vocab):
                if keep_vocab_item(word, v, min_count, trim_rule=trim_rule):
                    retain_words.append(word)
                    retain_total += v
                    if not dry_run:
                        self.wv.vocab[word] = Vocab(count=v, index=len(self.wv.index2word))
                        self.wv.index2word.append(word)
                else:
                    drop_unique += 1
                    drop_total += v
            original_unique_total = len(retain_words) + drop_unique
            retain_unique_pct = len(retain_words) * 100 / max(original_unique_total, 1)
            logger.info("min_count=%d retains %i unique words (%i%% of original %i, drops %i)",
                        min_count, len(retain_words), retain_unique_pct, original_unique_total, drop_unique)
            original_total = retain_total + drop_total
            retain_pct = retain_total * 100 / max(original_total, 1)
            logger.info("min_count=%d leaves %i word corpus (%i%% of original %i, drops %i)",
                        min_count, retain_total, retain_pct, original_total, drop_total)
        else:
            logger.info("Updating model with new vocabulary")
            new_total = pre_exist_total = 0
            new_words = pre_exist_words = []
            for word, v in iteritems(self.raw_vocab):
                if keep_vocab_item(word, v, min_count, trim_rule=trim_rule):
                    if word in self.wv.vocab:
                        pre_exist_words.append(word)
                        pre_exist_total += v
                        if not dry_run:
                            self.wv.vocab[word].count += v
                    else:
                        new_words.append(word)
                        new_total += v
                        if not dry_run:
                            self.wv.vocab[word] = Vocab(count=v, index=len(self.wv.index2word))
                            self.wv.index2word.append(word)
                else:
                    drop_unique += 1
                    drop_total += v
            original_unique_total = len(pre_exist_words) + len(new_words) + drop_unique
            pre_exist_unique_pct = len(pre_exist_words) * 100 / max(original_unique_total, 1)
            new_unique_pct = len(new_words) * 100 / max(original_unique_total, 1)
            logger.info("""New added %i unique words (%i%% of original %i)
                        and increased the count of %i pre-existing words (%i%% of original %i)""",
                        len(new_words), new_unique_pct, original_unique_total,
                        len(pre_exist_words), pre_exist_unique_pct, original_unique_total)
            retain_words = new_words + pre_exist_words
            retain_total = new_total + pre_exist_total

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = self.raw_vocab[w]
            word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                self.wv.vocab[w].sample_int = int(round(word_probability * 2**32))

        if not dry_run and not keep_raw_vocab:
            logger.info("deleting the raw counts dictionary of %i items", len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        logger.info("sample=%g downsamples %i most-common words", sample, downsample_unique)
        logger.info("downsampling leaves estimated %i word corpus (%.1f%% of prior %i)",
                    downsample_total, downsample_total * 100.0 / max(retain_total, 1), retain_total)

        # return from each step: words-affected, resulting-corpus-size
        report_values = {'drop_unique': drop_unique, 'retain_total': retain_total,
                         'downsample_unique': downsample_unique, 'downsample_total': int(downsample_total)}

        # print extra memory estimates
        report_values['memory'] = self.estimate_memory(vocab_size=len(retain_words))

        return report_values

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
        self.docvecs.clear_sims()

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

        self.docvecs.reset_weights(self)

    def reset_from(self, other_model):
        """
        from doc2vec
        """
        """Reuse shareable structures from other_model."""
        self.docvecs.borrow_from(other_model.docvecs)

        """
        from word2vec
        """
        self.wv.vocab = other_model.vocab
        self.wv.index2word = other_model.index2word
        self.cum_table = other_model.cum_table
        self.corpus_count = other_model.corpus_count
        self.reset_weights()

    def scan_vocab(self, documents, progress_per=10000, trim_rule=None, update=False):
        logger.info("collecting all words and their counts")
        document_no = -1
        total_words = 0
        min_reduce = 1
        interval_start = default_timer() - 0.00001  # guard against next sample being identical
        interval_count = 0
        checked_string_types = 0
        vocab = defaultdict(int)
        for document_no, document in enumerate(documents):
            if not checked_string_types:
                if isinstance(document.words, string_types):
                    logger.warn("Each 'words' should be a list of words (usually unicode strings)."
                                "First 'words' here is instead plain %s." % type(document.words))
                checked_string_types += 1
            if document_no % progress_per == 0:
                interval_rate = (total_words - interval_count) / (default_timer() - interval_start)
                logger.info("PROGRESS: at example #%i, processed %i words (%i/s), %i word types, %i tags",
                            document_no, total_words, interval_rate, len(vocab), len(self.docvecs))
                interval_start = default_timer()
                interval_count = total_words
            document_length = len(document.words)

            for tag in document.tags:
                self.docvecs.note_doctag(tag, document_no, document_length)

            for word in document.words:
                vocab[word] += 1
            total_words += len(document.words)

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1

        logger.info("collected %i word types and %i unique tags from a corpus of %i examples and %i words",
                    len(vocab), len(self.docvecs), document_no + 1, total_words)
        self.corpus_count = document_no + 1
        self.raw_vocab = vocab

    def _do_train_job(self, job, alpha, inits):
        work, neu1 = inits
        tally = 0
        for doc in job:
            indexed_doctags = self.docvecs.indexed_doctags(doc.tags)
            doctag_indexes, doctag_vectors, doctag_locks, ignored = indexed_doctags
            if self.sg:
                tally += train_document_dbow(self, doc.words, doctag_indexes, alpha, work,
                                             train_words=self.dbow_words,
                                             doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            elif self.dm_concat:
                tally += train_document_dm_concat(self, doc.words, doctag_indexes, alpha, work, neu1,
                                                  doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            else:
                tally += train_document_dm(self, doc.words, doctag_indexes, alpha, work, neu1,
                                           doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            self.docvecs.trained_item(indexed_doctags)
        return tally, self._raw_word_count(job)

    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        return sum(len(sentence.words) for sentence in job)

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
        report['doctag_lookup'] = self.docvecs.estimated_lookup_memory()
        report['doctag_syn0'] = self.docvecs.count * self.vector_size * dtype(REAL).itemsize


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
    """
    from word2vec
    """

    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        return sum(len(sentence) for sentence in job)

    def train(self, sentences, total_words=None, word_count=0,
              total_examples=None, queue_factor=2, report_delay=1.0):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For Word2Vec, each sentence must be a list of unicode strings. (Subclasses may accept other examples.)

        To support linear learning-rate decay from (initial) alpha to min_alpha, either total_examples
        (count of sentences) or total_words (count of raw words in sentences) should be provided, unless the
        sentences are the same as those that were used to initially build the vocabulary.

        """
        if (self.model_trimmed_post_training):
            raise RuntimeError("Parameters for training were discarded using model_trimmed_post_training method")
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension not loaded for Word2Vec, training will be slow. "
                          "Install a C compiler and reinstall gensim for fast training.")
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
                "The number of sentences in the training corpus is missing. Did you load the model via load_word2vec_format?"
                "Models loaded via load_word2vec_format don't support further training. "
                "Instead start with a blank model, scan_vocab on the new corpus, intersect_word2vec_format with the old model, then train.")

        if total_words is None and total_examples is None:
            if self.corpus_count:
                total_examples = self.corpus_count
                logger.info("expecting %i sentences, matching count from corpus used for vocabulary survey", total_examples)
            else:
                raise ValueError("you must provide either total_words or total_examples, to enable alpha and progress calculations")

        job_tally = 0

        if self.iter > 1:
            sentences = utils.RepeatCorpusNTimes(sentences, self.iter)
            total_words = total_words and total_words * self.iter
            total_examples = total_examples and total_examples * self.iter

        def worker_loop():
            """Train the model, lifting lists of sentences from the job_queue."""
            work = matutils.zeros_aligned(self.layer1_size, dtype=REAL)  # per-thread private work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            jobs_processed = 0
            while True:
                job = job_queue.get()
                if job is None:
                    progress_queue.put(None)
                    break  # no more jobs => quit this worker
                sentences, alpha = job
                tally, raw_tally = self._do_train_job(sentences, alpha, (work, neu1))
                progress_queue.put((len(sentences), tally, raw_tally))  # report back progress
                jobs_processed += 1
            logger.debug("worker exiting, processed %i jobs", jobs_processed)

        def job_producer():
            """Fill jobs queue using the input `sentences` iterator."""
            job_batch, batch_size = [], 0
            pushed_words, pushed_examples = 0, 0
            next_alpha = self.alpha
            if next_alpha > self.min_alpha_yet_reached:
                logger.warn("Effective 'alpha' higher than previous training cycles")
            self.min_alpha_yet_reached = next_alpha
            job_no = 0

            for sent_idx, sentence in enumerate(sentences):
                sentence_length = self._raw_word_count([sentence])

                # can we fit this sentence into the existing job batch?
                if batch_size + sentence_length <= self.batch_words:
                    # yes => add it to the current job
                    job_batch.append(sentence)
                    batch_size += sentence_length
                else:
                    # no => submit the existing job
                    logger.debug(
                        "queueing job #%i (%i words, %i sentences) at alpha %.05f",
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
                            pushed_words += self._raw_word_count(job_batch)
                            progress = 1.0 * pushed_words / total_words
                        next_alpha = self.alpha - (self.alpha - self.min_alpha) * progress
                        next_alpha = max(self.min_alpha, next_alpha)

                    # add the sentence that didn't fit as the first item of a new job
                    job_batch, batch_size = [sentence], sentence_length

            # add the last job too (may be significantly smaller than batch_words)
            if job_batch:
                logger.debug(
                    "queueing job #%i (%i words, %i sentences) at alpha %.05f",
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
        """
        Discard parameters that are used in training and score. Use if you're sure you're done training a model.
        Set `keep_doctags_vectors` to False if you don't want to save doctags vectors,
        in this case you can't to use docvecs's most_similar, similarity etc. methods.
        Set `keep_inference` to False if you don't want to store parameters that is used for infer_vector method
        """
        if not keep_inference:
            self._minimize_model(False, False, False)
        if self.docvecs and hasattr(self.docvecs, 'doctag_syn0') and not keep_doctags_vectors:
            del self.docvecs.doctag_syn0
        if self.docvecs and hasattr(self.docvecs, 'doctag_syn0_lockf'):
            del self.docvecs.doctag_syn0_lockf
