/* ==================================================================================================
 * engram_rng.c -- xoshiro256** with splitmix64 seeding. Integer arithmetic only.
 * ============================================================================================== */
#include "engram_rng.h"

static uint64_t engram_rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

/* splitmix64 as a STATEFUL stream, used only to expand one seed into four state words. Distinct from
 * engram_mix64 (which is its stateless finaliser) only in that it advances. */
static uint64_t engram_splitmix_next(uint64_t *x)
{
    uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void engram_rng_seed(engram_rng *r, uint64_t seed)
{
    uint64_t x = seed;
    if (!r) return;
    r->s[0] = engram_splitmix_next(&x);
    r->s[1] = engram_splitmix_next(&x);
    r->s[2] = engram_splitmix_next(&x);
    r->s[3] = engram_splitmix_next(&x);
    /* All-zero state is xoshiro's one fixed point. splitmix64 cannot produce four zeros in a row from
     * any seed, but the guard costs nothing and removes the question. */
    if (!(r->s[0] | r->s[1] | r->s[2] | r->s[3])) r->s[0] = 1u;
}

uint64_t engram_rng_u64(engram_rng *r)
{
    uint64_t result, t;
    result = engram_rotl(r->s[1] * 5u, 7) * 9u;
    t = r->s[1] << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = engram_rotl(r->s[3], 45);
    return result;
}

/* The HIGH half: xoshiro's upper bits are its strongest. */
uint32_t engram_rng_u32(engram_rng *r) { return (uint32_t)(engram_rng_u64(r) >> 32); }

uint64_t engram_rng_below(engram_rng *r, uint64_t n)
{
    uint64_t threshold, x;
    if (n == 0) return 0;
    /* (2^64 - n) mod n: the count of low values that would be over-represented by a bare modulo.
     * Rejecting them leaves an exact multiple of n equally likely values. Expected draws < 2. */
    threshold = (0u - n) % n;
    do { x = engram_rng_u64(r); } while (x < threshold);
    return x % n;
}

double engram_rng_unit(engram_rng *r)
{
    return (double)(engram_rng_u64(r) >> 11) * (1.0 / 9007199254740992.0);      /* 2^-53 */
}

float engram_rng_unitf(engram_rng *r)
{
    return (float)(engram_rng_u64(r) >> 40) * (1.0f / 16777216.0f);             /* 2^-24 */
}

double engram_rng_range(engram_rng *r, double lo, double hi)
{
    return lo + (hi - lo) * engram_rng_unit(r);
}

double engram_rng_normal(engram_rng *r)
{
    double s = 0.0;
    int i;
    for (i = 0; i < 12; i++) s += engram_rng_unit(r);
    return s - 6.0;
}

void engram_rng_shuffle_u32(engram_rng *r, uint32_t *a, size_t n)
{
    size_t i;
    if (!r || !a || n < 2u) return;
    for (i = n - 1u; i > 0u; i--) {
        size_t j = (size_t)engram_rng_below(r, (uint64_t)i + 1u);
        uint32_t t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}

void engram_rng_jump(engram_rng *r)
{
    static const uint64_t JUMP[4] = {
        0x180EC6D33CFD0ABAull, 0xD5A61266F0C9392Cull,
        0xA9582618E03FC9AAull, 0x39ABDC4529B1661Cull
    };
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i, b;
    for (i = 0; i < 4; i++) {
        for (b = 0; b < 64; b++) {
            if (JUMP[i] & ((uint64_t)1u << b)) {
                s0 ^= r->s[0]; s1 ^= r->s[1]; s2 ^= r->s[2]; s3 ^= r->s[3];
            }
            (void)engram_rng_u64(r);
        }
    }
    r->s[0] = s0; r->s[1] = s1; r->s[2] = s2; r->s[3] = s3;
}
