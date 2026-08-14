/*
 * bwcore — native kernels for Brain Workshop.
 *
 * These routines replace the hottest Python loops: Jaeggi/BT sequence
 * construction, session scoring, graph aggregation, stats-file parsing,
 * variable n-back draws, and rounded-rectangle vertex generation.
 *
 * Built as the ``bwcore`` CPython extension (see setup.py).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int bw_lround_int(double x)
{
    return (int)floor(x + 0.5);
}

/* -------------------------------------------------------------------------- */
/* Fast PCG32 RNG                                                             */
/* -------------------------------------------------------------------------- */

typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_t;

static pcg32_t g_rng;

static uint32_t pcg32(void)
{
    uint64_t old = g_rng.state;
    g_rng.state = old * 6364136223846793005ULL + g_rng.inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-(int32_t)rot) & 31));
}

static void pcg32_seed(uint64_t seed)
{
    g_rng.state = 0U;
    g_rng.inc = (seed << 1u) | 1u;
    (void)pcg32();
    g_rng.state += seed;
    (void)pcg32();
}

static void pcg32_seed_auto(void)
{
    uint64_t seed = (uint64_t)time(NULL);
    seed ^= (uint64_t)(uintptr_t)&g_rng;
    seed ^= ((uint64_t)clock() << 32);
#if defined(__APPLE__) || defined(__linux__)
    {
        FILE *ur = fopen("/dev/urandom", "rb");
        if (ur) {
            uint64_t extra = 0;
            if (fread(&extra, sizeof extra, 1, ur) == 1)
                seed ^= extra;
            fclose(ur);
        }
    }
#endif
    pcg32_seed(seed ? seed : 0x853c49e6748fea9bULL);
}

/* Inclusive integer in [lo, hi]. Unbiased for ranges that fit in 32 bits. */
static int randint_incl(int lo, int hi)
{
    uint32_t span;
    uint32_t threshold;
    uint32_t r;

    if (hi <= lo)
        return lo;
    span = (uint32_t)(hi - lo + 1);
    threshold = (uint32_t)(-span) % span;
    do {
        r = pcg32();
    } while (r < threshold);
    return lo + (int)(r % span);
}

static int rand_1_8(void)
{
    return 1 + (int)(pcg32() & 7u);
}

static double rand_u01(void)
{
    /* 24-bit mantissa in (0, 1] — never exactly 0, so pow() is safe. */
    return ((pcg32() >> 8) + 1u) * (1.0 / 16777216.0);
}

/* -------------------------------------------------------------------------- */
/* Small helpers                                                              */
/* -------------------------------------------------------------------------- */

static void shuffle_ints(int *a, int n)
{
    int i, j, tmp;
    for (i = n - 1; i > 0; --i) {
        j = randint_incl(0, i);
        tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }
}

static int crab_back(int x, int nback)
{
    return 1 + 2 * (x % nback);
}

static int resolve_back(int x, int nback, int crab,
                        const int *varlist, Py_ssize_t varlen)
{
    int back = crab ? crab_back(x, nback) : nback;
    if (varlist) {
        Py_ssize_t idx = (Py_ssize_t)x - (Py_ssize_t)back;
        if (idx >= 0 && idx < varlen)
            back = varlist[idx];
    }
    if (back < 1)
        back = 1;
    return back;
}

static int *list_to_ints(PyObject *obj, Py_ssize_t *out_n)
{
    Py_ssize_t i, n;
    int *buf;

    if (obj == NULL || obj == Py_None) {
        *out_n = 0;
        return NULL;
    }
    if (!PySequence_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "expected a sequence of integers");
        return NULL;
    }
    n = PySequence_Size(obj);
    if (n < 0)
        return NULL;
    buf = (int *)PyMem_Malloc((size_t)n * sizeof(int));
    if (!buf) {
        PyErr_NoMemory();
        return NULL;
    }
    for (i = 0; i < n; ++i) {
        PyObject *item = PySequence_GetItem(obj, i);
        long v;
        if (!item) {
            PyMem_Free(buf);
            return NULL;
        }
        v = PyLong_AsLong(item);
        Py_DECREF(item);
        if (v == -1 && PyErr_Occurred()) {
            PyMem_Free(buf);
            return NULL;
        }
        buf[i] = (int)v;
    }
    *out_n = n;
    return buf;
}

static PyObject *ints_to_list(const int *a, int n)
{
    PyObject *lst = PyList_New(n);
    int i;
    if (!lst)
        return NULL;
    for (i = 0; i < n; ++i) {
        PyObject *v = PyLong_FromLong(a[i]);
        if (!v) {
            Py_DECREF(lst);
            return NULL;
        }
        PyList_SET_ITEM(lst, i, v);
    }
    return lst;
}

/* -------------------------------------------------------------------------- */
/* 1. Constructive Jaeggi / BT sequence                                       */
/*                                                                            */
/* Original Python used nested rejection sampling of fully-random sequences   */
/* until it hit exactly 6 position matches, 6 audio matches and 2 duals.      */
/* That is O(lucky) and becomes unusable at high n (trials ≈ 20 + n²).        */
/*                                                                            */
/* Here we *construct* a sequence with those exact match counts in O(T):      */
/* pick which trials are both / position-only / audio-only / neither, then    */
/* fill values 1–8 honoring those constraints.                                */
/* -------------------------------------------------------------------------- */

static int build_bt_sequence(int ntrials, int nback,
                             int n_pos, int n_audio, int n_both,
                             int *pos, int *audio)
{
    int T, n_pos_only, n_aud_only, n_neither;
    int *kind = NULL;
    int i, t;

    if (ntrials < 1 || nback < 1 || nback >= ntrials)
        return -1;

    T = ntrials - nback;
    if (n_both < 0 || n_pos < n_both || n_audio < n_both)
        return -1;
    n_pos_only = n_pos - n_both;
    n_aud_only = n_audio - n_both;
    n_neither = T - n_pos_only - n_aud_only - n_both;
    if (n_neither < 0)
        return -1;

    kind = (int *)PyMem_Malloc((size_t)T * sizeof(int));
    if (!kind)
        return -2;

    i = 0;
    for (t = 0; t < n_both; ++t)
        kind[i++] = 3; /* both */
    for (t = 0; t < n_pos_only; ++t)
        kind[i++] = 1; /* position only */
    for (t = 0; t < n_aud_only; ++t)
        kind[i++] = 2; /* audio only */
    for (t = 0; t < n_neither; ++t)
        kind[i++] = 0; /* neither */
    shuffle_ints(kind, T);

    for (t = 0; t < nback; ++t) {
        pos[t] = rand_1_8();
        audio[t] = rand_1_8();
    }

    for (t = 0; t < T; ++t) {
        int idx = t + nback;
        int k = kind[t];
        if (k & 1) {
            pos[idx] = pos[idx - nback];
        } else {
            pos[idx] = rand_1_8();
            if (pos[idx] == pos[idx - nback])
                pos[idx] = (pos[idx] == 8) ? 1 : pos[idx] + 1;
        }
        if (k & 2) {
            audio[idx] = audio[idx - nback];
        } else {
            audio[idx] = rand_1_8();
            if (audio[idx] == audio[idx - nback])
                audio[idx] = (audio[idx] == 8) ? 1 : audio[idx] + 1;
        }
    }

    PyMem_Free(kind);
    return 0;
}

static PyObject *py_compute_bt_sequence(PyObject *self, PyObject *args)
{
    int ntrials, nback;
    int n_pos = 6, n_audio = 6, n_both = 2;
    int *pos = NULL, *audio = NULL;
    PyObject *pos_list = NULL, *audio_list = NULL, *result = NULL;
    int rc;

    (void)self;
    if (!PyArg_ParseTuple(args, "ii|iii", &ntrials, &nback, &n_pos, &n_audio, &n_both))
        return NULL;

    if (ntrials <= 0 || nback <= 0 || nback >= ntrials) {
        PyErr_SetString(PyExc_ValueError,
                        "num_trials must be > nback and both must be positive");
        return NULL;
    }

    pos = (int *)PyMem_Malloc((size_t)ntrials * sizeof(int));
    audio = (int *)PyMem_Malloc((size_t)ntrials * sizeof(int));
    if (!pos || !audio) {
        PyMem_Free(pos);
        PyMem_Free(audio);
        return PyErr_NoMemory();
    }

    rc = build_bt_sequence(ntrials, nback, n_pos, n_audio, n_both, pos, audio);
    if (rc != 0) {
        PyMem_Free(pos);
        PyMem_Free(audio);
        if (rc == -2)
            return PyErr_NoMemory();
        PyErr_SetString(PyExc_ValueError,
                        "cannot realize requested match counts with this trial/n-back");
        return NULL;
    }

    pos_list = ints_to_list(pos, ntrials);
    audio_list = ints_to_list(audio, ntrials);
    PyMem_Free(pos);
    PyMem_Free(audio);
    if (!pos_list || !audio_list) {
        Py_XDECREF(pos_list);
        Py_XDECREF(audio_list);
        return NULL;
    }
    result = Py_BuildValue("[NN]", pos_list, audio_list);
    return result;
}

/* -------------------------------------------------------------------------- */
/* 2. Session analysis (rights / wrongs per modality)                         */
/* -------------------------------------------------------------------------- */

enum {
    KIND_DIRECT = 0, /* compare session[mod][x] vs session[mod][x-back] */
    KIND_COMBO = 1,  /* visvis / visaudio / audiovis */
    KIND_SKIP = 2    /* arithmetic — handled in Python (Decimal) */
};

static int modality_kind(const char *name)
{
    if (strcmp(name, "visvis") == 0 ||
        strcmp(name, "visaudio") == 0 ||
        strcmp(name, "audiovis") == 0)
        return KIND_COMBO;
    if (strcmp(name, "arithmetic") == 0)
        return KIND_SKIP;
    return KIND_DIRECT;
}

static void combo_keys(const char *mod, const char **now_key, const char **then_key)
{
    /* visvis: vis vs vis; visaudio: vis vs audio; audiovis: audio vs vis */
    if (strncmp(mod, "vis", 3) == 0)
        *now_key = "vis";
    else
        *now_key = "audio";
    {
        size_t n = strlen(mod);
        if (n >= 3 && strcmp(mod + n - 3, "vis") == 0)
            *then_key = "vis";
        else
            *then_key = "audio";
    }
}

static void score_direct(const int *data, const int *inp, Py_ssize_t n,
                         int nback, int crab, int jaeggi,
                         const int *varlist, Py_ssize_t varlen,
                         int *rights, int *wrongs)
{
    Py_ssize_t x;
    *rights = 0;
    *wrongs = 0;
    for (x = nback; x < n; ++x) {
        int back = resolve_back((int)x, nback, crab, varlist, varlen);
        int match, inpv;
        if (back > (int)x)
            continue;
        match = (data[x] == data[x - back]);
        inpv = inp ? inp[x] : 0;
        *rights += (match && inpv);
        *wrongs += (match ^ inpv);
        if (jaeggi)
            *rights += ((!match) && (!inpv));
    }
}

static void score_combo(const int *now, const int *thenv, const int *inp,
                        Py_ssize_t n, int nback, int crab, int jaeggi,
                        const int *varlist, Py_ssize_t varlen,
                        int *rights, int *wrongs)
{
    Py_ssize_t x;
    *rights = 0;
    *wrongs = 0;
    for (x = nback; x < n; ++x) {
        int back = resolve_back((int)x, nback, crab, varlist, varlen);
        int match, inpv;
        if (back > (int)x)
            continue;
        match = (now[x] == thenv[x - back]);
        inpv = inp ? inp[x] : 0;
        *rights += (match && inpv);
        *wrongs += (match ^ inpv);
        if (jaeggi)
            *rights += ((!match) && (!inpv));
    }
}

static PyObject *py_analyze_session(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int nback, crab = 0, jaeggi = 0;
    PyObject *var_obj = Py_None;
    PyObject *mods_obj = NULL;
    PyObject *session_obj = NULL;
    int *varlist = NULL;
    Py_ssize_t varlen = 0;
    Py_ssize_t mi, nmods;
    PyObject *result = NULL;
    static char *kwlist[] = {
        "nback", "crab", "jaeggi_scoring", "variable_list",
        "modalities", "session", NULL
    };

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|ppOOO", kwlist,
                                     &nback, &crab, &jaeggi,
                                     &var_obj, &mods_obj, &session_obj))
        return NULL;

    if (!mods_obj || !session_obj) {
        PyErr_SetString(PyExc_TypeError, "modalities and session are required");
        return NULL;
    }
    if (nback < 1) {
        PyErr_SetString(PyExc_ValueError, "nback must be >= 1");
        return NULL;
    }
    if (!PySequence_Check(mods_obj) || !PyDict_Check(session_obj)) {
        PyErr_SetString(PyExc_TypeError, "modalities must be a sequence and session a dict");
        return NULL;
    }

    if (var_obj && var_obj != Py_None) {
        varlist = list_to_ints(var_obj, &varlen);
        if (!varlist && varlen == 0 && PyErr_Occurred())
            return NULL;
    }

    nmods = PySequence_Size(mods_obj);
    if (nmods < 0) {
        PyMem_Free(varlist);
        return NULL;
    }

    result = PyDict_New();
    if (!result) {
        PyMem_Free(varlist);
        return NULL;
    }

    for (mi = 0; mi < nmods; ++mi) {
        PyObject *mobj = PySequence_GetItem(mods_obj, mi);
        const char *mod;
        int kind;
        int rights = 0, wrongs = 0;
        PyObject *tuple;

        if (!mobj) {
            Py_DECREF(result);
            PyMem_Free(varlist);
            return NULL;
        }
        if (!PyUnicode_Check(mobj)) {
            Py_DECREF(mobj);
            Py_DECREF(result);
            PyMem_Free(varlist);
            PyErr_SetString(PyExc_TypeError, "modality names must be strings");
            return NULL;
        }
        mod = PyUnicode_AsUTF8(mobj);
        kind = modality_kind(mod);

        if (kind == KIND_SKIP) {
            /* Leave arithmetic to Python. Signal with None. */
            if (PyDict_SetItem(result, mobj, Py_None) < 0) {
                Py_DECREF(mobj);
                Py_DECREF(result);
                PyMem_Free(varlist);
                return NULL;
            }
            Py_DECREF(mobj);
            continue;
        }

        if (kind == KIND_DIRECT) {
            char inkey[64];
            PyObject *dobj, *iobj;
            int *data = NULL, *inp = NULL;
            Py_ssize_t dn = 0, in = 0;
            Py_ssize_t n;

            dobj = PyDict_GetItemString(session_obj, mod); /* borrowed */
            PyOS_snprintf(inkey, sizeof inkey, "%s_input", mod);
            iobj = PyDict_GetItemString(session_obj, inkey);
            if (!dobj) {
                Py_DECREF(mobj);
                continue;
            }
            data = list_to_ints(dobj, &dn);
            if (!data && PyErr_Occurred()) {
                Py_DECREF(mobj);
                Py_DECREF(result);
                PyMem_Free(varlist);
                return NULL;
            }
            if (iobj) {
                inp = list_to_ints(iobj, &in);
                if (!inp && PyErr_Occurred()) {
                    PyMem_Free(data);
                    Py_DECREF(mobj);
                    Py_DECREF(result);
                    PyMem_Free(varlist);
                    return NULL;
                }
            }
            n = dn;
            if (inp && in < n)
                n = in;
            score_direct(data, inp, n, nback, crab, jaeggi, varlist, varlen,
                         &rights, &wrongs);
            PyMem_Free(data);
            PyMem_Free(inp);
        } else { /* KIND_COMBO */
            const char *now_key, *then_key;
            char inkey[64];
            PyObject *now_obj, *then_obj, *iobj;
            int *nowv = NULL, *thenv = NULL, *inp = NULL;
            Py_ssize_t nn = 0, tn = 0, in = 0, n;

            combo_keys(mod, &now_key, &then_key);
            now_obj = PyDict_GetItemString(session_obj, now_key);
            then_obj = PyDict_GetItemString(session_obj, then_key);
            PyOS_snprintf(inkey, sizeof inkey, "%s_input", mod);
            iobj = PyDict_GetItemString(session_obj, inkey);
            if (!now_obj || !then_obj) {
                Py_DECREF(mobj);
                continue;
            }
            nowv = list_to_ints(now_obj, &nn);
            thenv = list_to_ints(then_obj, &tn);
            if ((!nowv || !thenv) && PyErr_Occurred()) {
                PyMem_Free(nowv);
                PyMem_Free(thenv);
                Py_DECREF(mobj);
                Py_DECREF(result);
                PyMem_Free(varlist);
                return NULL;
            }
            if (iobj) {
                inp = list_to_ints(iobj, &in);
                if (!inp && PyErr_Occurred()) {
                    PyMem_Free(nowv);
                    PyMem_Free(thenv);
                    Py_DECREF(mobj);
                    Py_DECREF(result);
                    PyMem_Free(varlist);
                    return NULL;
                }
            }
            n = nn < tn ? nn : tn;
            if (inp && in < n)
                n = in;
            score_combo(nowv, thenv, inp, n, nback, crab, jaeggi,
                        varlist, varlen, &rights, &wrongs);
            PyMem_Free(nowv);
            PyMem_Free(thenv);
            PyMem_Free(inp);
        }

        tuple = Py_BuildValue("(ii)", rights, wrongs);
        if (!tuple || PyDict_SetItem(result, mobj, tuple) < 0) {
            Py_XDECREF(tuple);
            Py_DECREF(mobj);
            Py_DECREF(result);
            PyMem_Free(varlist);
            return NULL;
        }
        Py_DECREF(tuple);
        Py_DECREF(mobj);
    }

    PyMem_Free(varlist);
    return result;
}

/* -------------------------------------------------------------------------- */
/* 3. Graph day-score aggregation                                             */
/* -------------------------------------------------------------------------- */

enum {
    STYLE_N = 0,
    STYLE_PCT,
    STYLE_N_DOT,
    STYLE_N_2PCT,
    STYLE_THRESH
};

static int style_from_obj(PyObject *obj, int *out)
{
    const char *s;
    if (PyLong_Check(obj)) {
        *out = (int)PyLong_AsLong(obj);
        return 0;
    }
    if (!PyUnicode_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "style must be int or str");
        return -1;
    }
    s = PyUnicode_AsUTF8(obj);
    if (strcmp(s, "N") == 0)
        *out = STYLE_N;
    else if (strcmp(s, "%") == 0)
        *out = STYLE_PCT;
    else if (strcmp(s, "N.%") == 0)
        *out = STYLE_N_DOT;
    else if (strcmp(s, "N+2*%-1") == 0)
        *out = STYLE_N_2PCT;
    else if (strcmp(s, "N+10/3+4/3") == 0)
        *out = STYLE_THRESH;
    else {
        PyErr_Format(PyExc_ValueError, "unknown graph style: %s", s);
        return -1;
    }
    return 0;
}

static PyObject *py_aggregate_day_scores(PyObject *self, PyObject *args)
{
    PyObject *style_obj, *entries;
    double adv = 80.0, flb = 50.0;
    int style;
    Py_ssize_t i, n;
    double sum = 0.0, mx = -1e300;
    double m = 0.0, b = 0.0;
    int count = 0;

    (void)self;
    if (!PyArg_ParseTuple(args, "OO|dd", &style_obj, &entries, &adv, &flb))
        return NULL;
    if (style_from_obj(style_obj, &style) < 0)
        return NULL;
    if (!PySequence_Check(entries)) {
        PyErr_SetString(PyExc_TypeError, "entries must be a sequence");
        return NULL;
    }
    if (style == STYLE_THRESH) {
        double den = adv - flb;
        if (den == 0.0)
            den = 1.0;
        m = 1.0 / den;
        b = -m * flb;
    }

    n = PySequence_Size(entries);
    if (n < 0)
        return NULL;

    for (i = 0; i < n; ++i) {
        PyObject *entry = PySequence_GetItem(entries, i);
        PyObject *n0, *n1;
        long nback, percent;
        double score;

        if (!entry)
            return NULL;
        if (!PySequence_Check(entry) || PySequence_Size(entry) < 2) {
            Py_DECREF(entry);
            PyErr_SetString(PyExc_TypeError, "each entry must be [nback, percent, ...]");
            return NULL;
        }
        n0 = PySequence_GetItem(entry, 0);
        n1 = PySequence_GetItem(entry, 1);
        Py_DECREF(entry);
        if (!n0 || !n1) {
            Py_XDECREF(n0);
            Py_XDECREF(n1);
            return NULL;
        }
        nback = PyLong_AsLong(n0);
        percent = PyLong_AsLong(n1);
        Py_DECREF(n0);
        Py_DECREF(n1);
        if ((nback == -1 || percent == -1) && PyErr_Occurred())
            return NULL;

        switch (style) {
        case STYLE_N:      score = (double)nback; break;
        case STYLE_PCT:    score = 0.01 * (double)percent; break;
        case STYLE_N_DOT:  score = (double)nback + 0.01 * (double)percent; break;
        case STYLE_N_2PCT: score = (double)nback - 1.0 + 2.0 * 0.01 * (double)percent; break;
        default:           score = (double)nback + b + m * (double)percent; break;
        }
        sum += score;
        if (score > mx)
            mx = score;
        ++count;
    }

    if (count == 0)
        return Py_BuildValue("(dd)", 0.0, 0.0);
    return Py_BuildValue("(dd)", sum / (double)count, mx);
}

/* -------------------------------------------------------------------------- */
/* 4. Rounded-rectangle vertices (old-style squares)                          */
/* -------------------------------------------------------------------------- */

static PyObject *py_rounded_rect_vertices(PyObject *self, PyObject *args)
{
    int lx, rx, by, ty, cr;
    int xy[80];
    int n = 0;
    int i;
    const double deg = M_PI / 180.0;

    (void)self;
    if (!PyArg_ParseTuple(args, "iiiii", &lx, &rx, &by, &ty, &cr))
        return NULL;

    /* x: BL cos 0..90, BR sin 0..90, TR sin 90..0, TL cos 90..0 */
    for (i = 0; i <= 90; i += 10)
        xy[n++] = lx + (int)(cr * (1.0 - cos(i * deg)));
    for (i = 0; i <= 90; i += 10)
        xy[n++] = rx - (int)(cr * (1.0 - sin(i * deg)));
    for (i = 90; i >= 0; i -= 10)
        xy[n++] = rx - (int)(cr * (1.0 - sin(i * deg)));
    for (i = 90; i >= 0; i -= 10)
        xy[n++] = lx + (int)(cr * (1.0 - cos(i * deg)));

    /* y is built as two 20-long blocks (0..90 then 90..0) */
    n = 0;
    {
        int ytmp[40];
        int k = 0;
        for (i = 0; i <= 90; i += 10)
            ytmp[k++] = by + (int)(cr * (1.0 - sin(i * deg)));
        for (i = 90; i >= 0; i -= 10)
            ytmp[k++] = by + (int)(cr * (1.0 - sin(i * deg)));
        for (i = 0; i <= 90; i += 10)
            ytmp[k++] = ty - (int)(cr * (1.0 - sin(i * deg)));
        for (i = 90; i >= 0; i -= 10)
            ytmp[k++] = ty - (int)(cr * (1.0 - sin(i * deg)));
        /* Interleave: we stored x in xy[0..39], now write pairs into a new list. */
        {
            PyObject *lst = PyList_New(80);
            int p;
            if (!lst)
                return NULL;
            /* x was written into xy[0..39] already; y is ytmp. Rebuild list. */
            /* Wait — we overwrote n and then reused xy only for x. xy[0..39] still holds x. */
            for (p = 0; p < 40; ++p) {
                PyObject *xv = PyLong_FromLong(xy[p]);
                PyObject *yv = PyLong_FromLong(ytmp[p]);
                if (!xv || !yv) {
                    Py_XDECREF(xv);
                    Py_XDECREF(yv);
                    Py_DECREF(lst);
                    return NULL;
                }
                PyList_SET_ITEM(lst, p * 2, xv);
                PyList_SET_ITEM(lst, p * 2 + 1, yv);
            }
            return lst;
        }
    }
}

/* -------------------------------------------------------------------------- */
/* 5. Variable n-back draws  (Beta(back/2, 1) = U^(2/back))                   */
/* -------------------------------------------------------------------------- */

static PyObject *py_variable_nback_list(PyObject *self, PyObject *args)
{
    int count, back;
    PyObject *lst;
    int i;
    double inv_alpha;

    (void)self;
    if (!PyArg_ParseTuple(args, "ii", &count, &back))
        return NULL;
    if (count < 0 || back < 1) {
        PyErr_SetString(PyExc_ValueError, "count >= 0 and back >= 1 required");
        return NULL;
    }
    lst = PyList_New(count);
    if (!lst)
        return NULL;
    inv_alpha = 2.0 / (double)back;
    for (i = 0; i < count; ++i) {
        double u = rand_u01();
        int v = (int)(pow(u, inv_alpha) * (double)back + 1.0);
        if (v < 1)
            v = 1;
        if (v > back)
            v = back;
        {
            PyObject *o = PyLong_FromLong(v);
            if (!o) {
                Py_DECREF(lst);
                return NULL;
            }
            PyList_SET_ITEM(lst, i, o);
        }
    }
    return lst;
}

/* -------------------------------------------------------------------------- */
/* 6. sample k distinct ints from [lo, hi] (Fisher–Yates)                     */
/* -------------------------------------------------------------------------- */

static PyObject *py_sample_unique(PyObject *self, PyObject *args)
{
    int lo, hi, k;
    int span, i;
    int *pool;
    PyObject *lst;

    (void)self;
    if (!PyArg_ParseTuple(args, "iii", &lo, &hi, &k))
        return NULL;
    if (hi < lo) {
        PyErr_SetString(PyExc_ValueError, "hi must be >= lo");
        return NULL;
    }
    span = hi - lo + 1;
    if (k < 0 || k > span) {
        PyErr_SetString(PyExc_ValueError, "k must be in [0, hi-lo+1]");
        return NULL;
    }
    pool = (int *)PyMem_Malloc((size_t)span * sizeof(int));
    if (!pool)
        return PyErr_NoMemory();
    for (i = 0; i < span; ++i)
        pool[i] = lo + i;
    for (i = 0; i < k; ++i) {
        int j = randint_incl(i, span - 1);
        int tmp = pool[i];
        pool[i] = pool[j];
        pool[j] = tmp;
    }
    lst = ints_to_list(pool, k);
    PyMem_Free(pool);
    return lst;
}

/* -------------------------------------------------------------------------- */
/* 7. Stats-file parser                                                       */
/*                                                                            */
/* Returns a list of dicts:                                                   */
/*   year, month, day, hour, minute, second,                                  */
/*   percent, mode, nback, ticks, trials, manual, session, sesstime,          */
/*   cats: 16-int list of category percents (indices 9..24 of the CSV)        */
/* -------------------------------------------------------------------------- */

static int parse_int_field(const char *s, int *out)
{
    char *end = NULL;
    long v;
    if (!s || !*s) {
        *out = 0;
        return 0;
    }
    v = strtol(s, &end, 10);
    if (end == s) {
        *out = 0;
        return 0;
    }
    *out = (int)v;
    return 1;
}

static int parse_double_round(const char *s, int *out)
{
    char *end = NULL;
    double v;
    if (!s || !*s) {
        *out = 0;
        return 0;
    }
    v = strtod(s, &end);
    if (end == s) {
        *out = 0;
        return 0;
    }
    *out = bw_lround_int(v);
    return 1;
}

static PyObject *parse_one_stats_line(const char *line, Py_ssize_t len)
{
    char buf[1024];
    char *fields[40];
    int nfields = 0;
    char sep;
    char *p;
    int y, mo, d, H, M, S;
    int percent, mode, nback, ticks, trials, manual, session, sesstime;
    int cats[16];
    int i;
    PyObject *dict, *catlist;

    if (len <= 0)
        return NULL;
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
        --len;
    if (len <= 0 || line[0] < '0' || line[0] > '9')
        return NULL;
    if ((Py_ssize_t)sizeof(buf) <= len)
        return NULL;
    memcpy(buf, line, (size_t)len);
    buf[len] = '\0';

    /* datetime prefix: YYYY-MM-DD HH:MM:SS */
    if (len < 19)
        return NULL;
    if (buf[4] != '-' || buf[7] != '-' || buf[10] != ' ' ||
        buf[13] != ':' || buf[16] != ':')
        return NULL;
    y  = (buf[0] - '0') * 1000 + (buf[1] - '0') * 100 + (buf[2] - '0') * 10 + (buf[3] - '0');
    mo = (buf[5] - '0') * 10 + (buf[6] - '0');
    d  = (buf[8] - '0') * 10 + (buf[9] - '0');
    H  = (buf[11] - '0') * 10 + (buf[12] - '0');
    M  = (buf[14] - '0') * 10 + (buf[15] - '0');
    S  = (buf[17] - '0') * 10 + (buf[18] - '0');

    /* remainder after datetime */
    p = buf + 19;
    while (*p == ' ')
        ++p;
    sep = strchr(p, '\t') ? '\t' : ',';

    /* split whole line so field indices match the original Python parser */
    p = buf;
    fields[nfields++] = p;
    while (*p && nfields < 40) {
        if (*p == sep) {
            *p = '\0';
            fields[nfields++] = p + 1;
        }
        ++p;
    }

    /* Need at least through session number (index 8). */
    if (nfields < 9)
        return NULL;

    parse_int_field(fields[2], &percent);
    parse_int_field(fields[3], &mode);
    parse_int_field(fields[4], &nback);
    parse_int_field(nfields > 5 ? fields[5] : "", &ticks);
    parse_int_field(nfields > 6 ? fields[6] : "", &trials);
    parse_int_field(fields[7], &manual);
    parse_int_field(fields[8], &session);

    memset(cats, 0, sizeof cats);
    for (i = 0; i < 16; ++i) {
        int idx = 9 + i;
        if (idx < nfields)
            parse_int_field(fields[idx], &cats[i]);
    }

    sesstime = 0;
    if (nfields > 25)
        parse_double_round(fields[25], &sesstime);

    dict = PyDict_New();
    if (!dict)
        return NULL;
    catlist = ints_to_list(cats, 16);
    if (!catlist) {
        Py_DECREF(dict);
        return NULL;
    }

#define SET_INT(k, v) do { \
        PyObject *_o = PyLong_FromLong(v); \
        if (!_o || PyDict_SetItemString(dict, k, _o) < 0) { \
            Py_XDECREF(_o); Py_DECREF(catlist); Py_DECREF(dict); return NULL; \
        } \
        Py_DECREF(_o); \
    } while (0)

    SET_INT("year", y);
    SET_INT("month", mo);
    SET_INT("day", d);
    SET_INT("hour", H);
    SET_INT("minute", M);
    SET_INT("second", S);
    SET_INT("percent", percent);
    SET_INT("mode", mode);
    SET_INT("nback", nback);
    SET_INT("ticks", ticks);
    SET_INT("trials", trials);
    SET_INT("manual", manual);
    SET_INT("session", session);
    SET_INT("sesstime", sesstime);
#undef SET_INT

    if (PyDict_SetItemString(dict, "cats", catlist) < 0) {
        Py_DECREF(catlist);
        Py_DECREF(dict);
        return NULL;
    }
    Py_DECREF(catlist);
    return dict;
}

static PyObject *py_parse_stats_text(PyObject *self, PyObject *args)
{
    const char *text;
    Py_ssize_t n = 0;
    Py_ssize_t i, start;
    PyObject *out;

    (void)self;
    if (!PyArg_ParseTuple(args, "s#", &text, &n))
        return NULL;

    out = PyList_New(0);
    if (!out)
        return NULL;

    start = 0;
    for (i = 0; i <= n; ++i) {
        if (i == n || text[i] == '\n') {
            PyObject *rec = parse_one_stats_line(text + start, i - start);
            if (rec) {
                if (PyList_Append(out, rec) < 0) {
                    Py_DECREF(rec);
                    Py_DECREF(out);
                    return NULL;
                }
                Py_DECREF(rec);
            } else if (PyErr_Occurred()) {
                Py_DECREF(out);
                return NULL;
            }
            start = i + 1;
        }
    }
    return out;
}

/* -------------------------------------------------------------------------- */
/* 8. Simple n-back equality check over a history buffer                      */
/* -------------------------------------------------------------------------- */

static PyObject *py_is_nback_match(PyObject *self, PyObject *args)
{
    int current, nback_trial;
    PyObject *hist_obj;
    int *hist = NULL;
    Py_ssize_t n = 0;
    int match;

    (void)self;
    if (!PyArg_ParseTuple(args, "iOi", &current, &hist_obj, &nback_trial))
        return NULL;
    hist = list_to_ints(hist_obj, &n);
    if (!hist && PyErr_Occurred())
        return NULL;
    if (nback_trial < 0 || nback_trial >= n) {
        PyMem_Free(hist);
        Py_RETURN_NONE; /* unknown / not enough history */
    }
    match = (current == hist[nback_trial]);
    PyMem_Free(hist);
    if (match)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/* -------------------------------------------------------------------------- */
/* 9. Mean of a numeric sequence (last N, used by graph percents / averages)  */
/* -------------------------------------------------------------------------- */

static PyObject *py_mean_tail(PyObject *self, PyObject *args)
{
    PyObject *seq;
    int tail = 0;
    Py_ssize_t n, start, i;
    double sum = 0.0;
    Py_ssize_t count = 0;

    (void)self;
    if (!PyArg_ParseTuple(args, "O|i", &seq, &tail))
        return NULL;
    if (!PySequence_Check(seq)) {
        PyErr_SetString(PyExc_TypeError, "expected a sequence");
        return NULL;
    }
    n = PySequence_Size(seq);
    if (n < 0)
        return NULL;
    if (n == 0)
        return PyFloat_FromDouble(0.0);
    if (tail > 0 && tail < n)
        start = n - tail;
    else
        start = 0;
    for (i = start; i < n; ++i) {
        PyObject *item = PySequence_GetItem(seq, i);
        double v;
        if (!item)
            return NULL;
        v = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (v == -1.0 && PyErr_Occurred())
            return NULL;
        sum += v;
        ++count;
    }
    if (count == 0)
        return PyFloat_FromDouble(0.0);
    return PyFloat_FromDouble(sum / (double)count);
}

/* -------------------------------------------------------------------------- */
/* seed / backend                                                             */
/* -------------------------------------------------------------------------- */

static PyObject *py_seed(PyObject *self, PyObject *args)
{
    unsigned long long s = 0;
    (void)self;
    if (!PyArg_ParseTuple(args, "|K", &s))
        return NULL;
    if (s == 0)
        pcg32_seed_auto();
    else
        pcg32_seed((uint64_t)s);
    Py_RETURN_NONE;
}

static PyObject *py_backend(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
    return PyUnicode_FromString("C");
}

/* -------------------------------------------------------------------------- */
/* Module definition                                                          */
/* -------------------------------------------------------------------------- */

static PyMethodDef BwcoreMethods[] = {
    {"compute_bt_sequence", py_compute_bt_sequence, METH_VARARGS,
     "compute_bt_sequence(num_trials, nback, pos=6, audio=6, both=2) -> [pos, audio]"},
    {"analyze_session", (PyCFunction)py_analyze_session, METH_VARARGS | METH_KEYWORDS,
     "analyze_session(nback, crab=False, jaeggi_scoring=False, variable_list=None, "
     "modalities=..., session=...) -> {mod: (rights, wrongs) or None}"},
    {"aggregate_day_scores", py_aggregate_day_scores, METH_VARARGS,
     "aggregate_day_scores(style, entries, advance=80, fallback=50) -> (mean, max)"},
    {"rounded_rect_vertices", py_rounded_rect_vertices, METH_VARARGS,
     "rounded_rect_vertices(lx, rx, by, ty, cr) -> [x0,y0,...] (80 ints)"},
    {"variable_nback_list", py_variable_nback_list, METH_VARARGS,
     "variable_nback_list(count, back) -> list[int]"},
    {"sample_unique", py_sample_unique, METH_VARARGS,
     "sample_unique(lo, hi, k) -> k distinct ints in [lo, hi]"},
    {"parse_stats_text", py_parse_stats_text, METH_VARARGS,
     "parse_stats_text(text) -> list[dict]"},
    {"is_nback_match", py_is_nback_match, METH_VARARGS,
     "is_nback_match(current, history, nback_trial) -> bool or None"},
    {"mean_tail", py_mean_tail, METH_VARARGS,
     "mean_tail(seq, tail=0) -> float  (tail=0 means the whole sequence)"},
    {"seed", py_seed, METH_VARARGS, "seed(n=0) — 0 reseeds from entropy"},
    {"backend", py_backend, METH_NOARGS, "Return 'C'"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef bwcoremodule = {
    PyModuleDef_HEAD_INIT,
    "bwcore",
    "C kernels for Brain Workshop hot loops.",
    -1,
    BwcoreMethods
};

PyMODINIT_FUNC PyInit_bwcore(void)
{
    pcg32_seed_auto();
    return PyModule_Create(&bwcoremodule);
}
