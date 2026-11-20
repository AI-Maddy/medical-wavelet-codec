/*
 * CALIC: Context-based, Adaptive, Lossless Image Codec
 * (X. Wu and N. Memon, IEEE Trans. Communications 45(4), 1997), for 8- to 16-bit
 * greyscale images. Reimplemented here because no public CALIC binary for >8-bit
 * images exists.
 *
 * Implemented as in the paper:
 *   - binary mode when the six causal neighbours take at most two values
 *     (ternary symbol: = s1, = s2, escape), context = which neighbours equal s1;
 *   - gradient-adjusted prediction (GAP) with thresholds 80 / 32 / 8;
 *   - error-energy estimator D = dh + dv + 2|e_w| quantised to 8 levels
 *     (thresholds 5, 15, 25, 42, 60, 85, 140);
 *   - texture pattern B (8 bits: N, W, NW, NE, NN, WW, 2N-NN, 2W-WW below the
 *     prediction) combined with floor(level/2) into 1,024 compound contexts for
 *     bias cancellation (error feedback) and sign flipping;
 *   - entropy coding of the remapped error conditioned on the 8 energy levels.
 * For bit depth b > 8, every gradient and energy threshold is multiplied by 2^(b-8).
 *
 * Difference from the 8-bit original: the remapped error is binarised (bit length
 * with an adaptive model per energy level, the two bits below the MSB adaptive,
 * the rest raw) and coded with an adaptive binary arithmetic coder, because a
 * 65,536-symbol adaptive m-ary model per context would not learn on these image sizes.
 *
 * usage: calic e in.pgm out.cal  |  calic d in.cal out.pgm
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- adaptive binary arithmetic coder (carry-less, 32-bit) ---------------- */
typedef struct { uint32_t x1, x2, x; FILE *f; } AC;
typedef struct { uint16_t p; uint16_t n; } Bit;   /* P(1) in 16 bits, adaptation count */

static void bit_init(Bit *b) { b->p = 32768; b->n = 0; }
static void bit_update(Bit *b, int y) {
    int lim = 255;
    int target = y ? 65535 : 0;
    int d = (target - (int)b->p) / (b->n + 2);
    int np = (int)b->p + d;
    if (np < 64) np = 64;
    if (np > 65535 - 64) np = 65535 - 64;
    b->p = (uint16_t)np;
    if (b->n < lim) b->n++;
}
static void ac_enc_init(AC *a, FILE *f) { a->x1 = 0; a->x2 = 0xffffffffu; a->f = f; }
static void ac_dec_init(AC *a, FILE *f) {
    a->x1 = 0; a->x2 = 0xffffffffu; a->f = f; a->x = 0;
    for (int i = 0; i < 4; i++) { int c = getc(f); a->x = (a->x << 8) | (c == EOF ? 0 : c); }
}
static inline uint32_t ac_mid(AC *a, uint16_t p) {
    return a->x1 + (uint32_t)(((uint64_t)(a->x2 - a->x1) * p) >> 16);
}
static void enc_bit(AC *a, Bit *b, int y) {
    uint32_t xm = ac_mid(a, b->p);
    if (y) a->x2 = xm; else a->x1 = xm + 1;
    bit_update(b, y);
    while (((a->x1 ^ a->x2) & 0xff000000u) == 0) {
        putc(a->x2 >> 24, a->f); a->x1 <<= 8; a->x2 = (a->x2 << 8) | 255;
    }
}
static int dec_bit(AC *a, Bit *b) {
    uint32_t xm = ac_mid(a, b->p);
    int y = a->x <= xm;
    if (y) a->x2 = xm; else a->x1 = xm + 1;
    bit_update(b, y);
    while (((a->x1 ^ a->x2) & 0xff000000u) == 0) {
        a->x1 <<= 8; a->x2 = (a->x2 << 8) | 255;
        int c = getc(a->f); a->x = (a->x << 8) | (c == EOF ? 0 : c);
    }
    return y;
}
static void ac_flush(AC *a) { for (int i = 0; i < 4; i++) { putc(a->x1 >> 24, a->f); a->x1 <<= 8; } }
static Bit RAW = {32768, 0};
static void enc_raw(AC *a, int y) { Bit r = RAW; r.n = 255; uint32_t xm = ac_mid(a, 32768); if (y) a->x2 = xm; else a->x1 = xm + 1;
    while (((a->x1 ^ a->x2) & 0xff000000u) == 0) { putc(a->x2 >> 24, a->f); a->x1 <<= 8; a->x2 = (a->x2 << 8) | 255; } (void)r; }
static int dec_raw(AC *a) { uint32_t xm = ac_mid(a, 32768); int y = a->x <= xm; if (y) a->x2 = xm; else a->x1 = xm + 1;
    while (((a->x1 ^ a->x2) & 0xff000000u) == 0) { a->x1 <<= 8; a->x2 = (a->x2 << 8) | 255; int c = getc(a->f); a->x = (a->x << 8) | (c == EOF ? 0 : c); }
    return y; }

/* ---------------- models ---------------- */
#define NLEV 8
#define MAXK 18
typedef struct {
    Bit klen[NLEV][MAXK + 1];        /* unary bit-length code per energy level */
    Bit sub[NLEV][MAXK + 1][4];      /* two bits below the MSB (tree of 3 nodes) */
    Bit bin[64][2];                  /* binary mode: ternary symbol as two binary decisions */
    long esum[1024]; int ecnt[1024]; /* bias cancellation per compound context */
} Model;

static void model_init(Model *m) {
    for (int i = 0; i < NLEV; i++) for (int k = 0; k <= MAXK; k++) {
        bit_init(&m->klen[i][k]); for (int j = 0; j < 4; j++) bit_init(&m->sub[i][k][j]);
    }
    for (int i = 0; i < 64; i++) { bit_init(&m->bin[i][0]); bit_init(&m->bin[i][1]); }
    memset(m->esum, 0, sizeof m->esum); memset(m->ecnt, 0, sizeof m->ecnt);
}

static void code_value(AC *a, Model *m, int lev, uint32_t v, int dec, uint32_t *out) {
    /* v >= 0: k = bit length (0 for v = 0), unary coded; then bits below the MSB */
    int k = 0;
    if (!dec) { uint32_t t = v; while (t) { k++; t >>= 1; } }
    int kk = 0;
    for (;;) {
        if (kk == MAXK) break;
        int stop;
        if (dec) stop = dec_bit(a, &m->klen[lev][kk]);
        else { stop = (kk == k); enc_bit(a, &m->klen[lev][kk], stop); }
        if (stop) break;
        kk++;
    }
    if (dec) k = kk;
    if (k <= 1) { if (dec) *out = (uint32_t)k; return; }
    uint32_t r = 1;
    int node = 1;
    for (int i = k - 2; i >= 0; i--) {
        int y;
        int adaptive = (k - 2 - i) < 2;
        if (dec) y = adaptive ? dec_bit(a, &m->sub[lev][k][node]) : dec_raw(a);
        else { y = (v >> i) & 1; if (adaptive) enc_bit(a, &m->sub[lev][k][node], y); else enc_raw(a, y); }
        if (adaptive) node = node * 2 + y;
        if (node > 3) node = 3;
        r = (r << 1) | (uint32_t)y;
    }
    if (dec) *out = r;
}

/* fold error e into [0, maxv] given prediction p (both directions bounded by the range) */
static uint32_t fold(int e, int p, int maxv) {
    int lo = p, hi = maxv - p, t = lo < hi ? lo : hi;
    if (e >= -t && e <= t) return e >= 0 ? 2u * e : 2u * (-e) - 1;
    return (uint32_t)(2 * t + (e > 0 ? e - t : -e - t)) ;
}
static int unfold(uint32_t v, int p, int maxv) {
    int lo = p, hi = maxv - p, t = lo < hi ? lo : hi;
    if (v <= (uint32_t)(2 * t)) return (v & 1) ? -(int)((v + 1) / 2) : (int)(v / 2);
    int d = (int)v - 2 * t + t;
    return hi > lo ? d : -d;
}

/* ---------------- image I/O ---------------- */
static int *read_pgm(const char *fn, int *W, int *H, int *maxv) {
    FILE *f = fopen(fn, "rb"); if (!f) { perror(fn); exit(1); }
    char mg[3] = {0}; if (fscanf(f, "%2s", mg) != 1 || strcmp(mg, "P5")) { fprintf(stderr, "not P5\n"); exit(1); }
    int c; int vals[3]; for (int i = 0; i < 3; i++) {
        do { c = getc(f); if (c == '#') while (c != '\n' && c != EOF) c = getc(f); } while (c == ' ' || c == '\n' || c == '\r' || c == '\t');
        ungetc(c, f); if (fscanf(f, "%d", &vals[i]) != 1) exit(1);
    }
    getc(f);
    *W = vals[0]; *H = vals[1]; *maxv = vals[2];
    int n = *W * *H; int *img = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        if (*maxv > 255) { int a = getc(f), b = getc(f); img[i] = (a << 8) | b; } else img[i] = getc(f);
    }
    fclose(f); return img;
}
static void write_pgm(const char *fn, int *img, int W, int H, int maxv) {
    FILE *f = fopen(fn, "wb"); fprintf(f, "P5\n%d %d\n%d\n", W, H, maxv);
    for (int i = 0; i < W * H; i++) { if (maxv > 255) { putc(img[i] >> 8, f); putc(img[i] & 255, f); } else putc(img[i], f); }
    fclose(f);
}

/* ---------------- the codec ---------------- */
static int iabs(int x) { return x < 0 ? -x : x; }

static void run(int *img, int W, int H, int maxv, AC *a, int dec) {
    int bits = 1; while ((1 << bits) <= maxv) bits++;
    int sc = bits > 8 ? 1 << (bits - 8) : 1;
    static const int QT[7] = {5, 15, 25, 42, 60, 85, 140};
    int qt[7]; for (int i = 0; i < 7; i++) qt[i] = QT[i] * sc;
    Model *m = calloc(1, sizeof(Model)); model_init(m);
    int *err = calloc((size_t)W * H, sizeof(int));     /* prediction errors, for 2|e_w| */
#define P(yy, xx) img[(yy) * W + (xx)]
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int w, ww, n, nn, ne, nw, nne;
        if (y == 0 && x == 0) { w = ww = n = nn = ne = nw = nne = maxv / 2; }
        else if (y == 0) { w = P(0, x - 1); ww = x > 1 ? P(0, x - 2) : w; n = nn = ne = nw = nne = w; }
        else {
            n = P(y - 1, x);
            w = x > 0 ? P(y, x - 1) : n;
            ww = x > 1 ? P(y, x - 2) : w;
            nw = x > 0 ? P(y - 1, x - 1) : n;
            ne = x < W - 1 ? P(y - 1, x + 1) : n;
            nn = y > 1 ? P(y - 2, x) : n;
            nne = (y > 1 && x < W - 1) ? P(y - 2, x + 1) : ne;
        }
        /* ---- binary mode ---- */
        int nb[6] = {w, n, nw, ne, ww, nn};
        int s1 = w, s2 = -1, two = 1;
        for (int i = 1; i < 6; i++) if (nb[i] != s1) { if (s2 < 0) s2 = nb[i]; else if (nb[i] != s2) { two = 0; break; } }
        if (two) {
            int t = 0; for (int i = 0; i < 6; i++) t = (t << 1) | (nb[i] == s1);
            int sym;    /* 0: = s1, 1: = s2, 2: escape */
            if (dec) {
                if (dec_bit(a, &m->bin[t][0])) sym = 0;
                else sym = (s2 >= 0 && dec_bit(a, &m->bin[t][1])) ? 1 : 2;
            } else {
                int v = P(y, x);
                sym = v == s1 ? 0 : (s2 >= 0 && v == s2) ? 1 : 2;
                enc_bit(a, &m->bin[t][0], sym == 0);
                if (sym != 0 && s2 >= 0) enc_bit(a, &m->bin[t][1], sym == 1);
            }
            if (sym < 2) { if (dec) P(y, x) = sym == 0 ? s1 : s2; err[y * W + x] = 0; continue; }
        }
        /* ---- GAP ---- */
        int dh = iabs(w - ww) + iabs(n - nw) + iabs(n - ne);
        int dv = iabs(w - nw) + iabs(n - nn) + iabs(ne - nne);
        int pred;
        if (dv - dh > 80 * sc) pred = w;
        else if (dh - dv > 80 * sc) pred = n;
        else {
            double p = (w + n) / 2.0 + (ne - nw) / 4.0;
            if (dv - dh > 32 * sc) p = (p + w) / 2.0;
            else if (dv - dh > 8 * sc) p = (3 * p + w) / 4.0;
            else if (dh - dv > 32 * sc) p = (p + n) / 2.0;
            else if (dh - dv > 8 * sc) p = (3 * p + n) / 4.0;
            pred = (int)(p >= 0 ? p + 0.5 : p - 0.5);
        }
        /* ---- contexts ---- */
        int ew = x > 0 ? err[y * W + x - 1] : (y > 0 ? err[(y - 1) * W + x] : 0);
        int D = dh + dv + 2 * iabs(ew);
        int lev = 0; while (lev < 7 && D > qt[lev]) lev++;
        int B = 0, tx[8] = {n, w, nw, ne, nn, ww, 2 * n - nn, 2 * w - ww};
        for (int i = 0; i < 8; i++) B = (B << 1) | (tx[i] < pred);
        int C = (B << 2) | (lev >> 1);
        /* ---- bias cancellation and sign flipping ---- */
        int bias = 0, flip = 0;
        if (m->ecnt[C]) {
            double mb = (double)m->esum[C] / m->ecnt[C];
            bias = (int)(mb >= 0 ? mb + 0.5 : mb - 0.5);
            flip = mb < 0;
        }
        int pc = pred + bias; if (pc < 0) pc = 0; if (pc > maxv) pc = maxv;
        int pf = flip ? maxv - pc : pc;          /* sign flip done on the folded alphabet */
        int v;
        if (dec) {
            uint32_t u; code_value(a, m, lev, 0, 1, &u);
            int e = unfold(u, pf, maxv);
            v = flip ? pc - e : pc + e;
            P(y, x) = v;
        } else {
            v = P(y, x);
            int e = flip ? pc - v : v - pc;
            code_value(a, m, lev, fold(e, pf, maxv), 0, NULL);
        }
        int e0 = v - pred;                        /* error of the unbiased GAP prediction */
        err[y * W + x] = v - pc;
        m->esum[C] += e0; m->ecnt[C]++;
        if (m->ecnt[C] >= 255) { m->esum[C] /= 2; m->ecnt[C] /= 2; }
    }
#undef P
    free(m); free(err);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: calic e|d in out\n"); return 2; }
    if (argv[1][0] == 'e') {
        int W, H, maxv; int *img = read_pgm(argv[2], &W, &H, &maxv);
        FILE *f = fopen(argv[3], "wb");
        fwrite("CAL1", 1, 4, f);
        uint32_t hd[3] = {(uint32_t)W, (uint32_t)H, (uint32_t)maxv}; fwrite(hd, 4, 3, f);
        AC a; ac_enc_init(&a, f); run(img, W, H, maxv, &a, 0); ac_flush(&a);
        fclose(f); free(img);
    } else {
        FILE *f = fopen(argv[2], "rb"); char mg[4];
        if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "CAL1", 4)) { fprintf(stderr, "bad stream\n"); return 1; }
        uint32_t hd[3]; if (fread(hd, 4, 3, f) != 3) return 1;
        int W = hd[0], H = hd[1], maxv = hd[2];
        int *img = calloc((size_t)W * H, sizeof(int));
        AC a; ac_dec_init(&a, f); run(img, W, H, maxv, &a, 1);
        fclose(f); write_pgm(argv[3], img, W, H, maxv); free(img);
    }
    return 0;
}
