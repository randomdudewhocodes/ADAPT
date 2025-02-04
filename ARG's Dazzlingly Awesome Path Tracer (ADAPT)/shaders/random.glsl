uvec4 seed;

uvec4 pcg4d(uvec4 v)
{
    v = v * 1664525u + 1013904223u;
    
    v.x += v.y*v.w;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v.w += v.y*v.z;
    
    v ^= v >> 16u;
    
    v.x += v.y*v.w;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v.w += v.y*v.z;
    
    return v;
}

uvec4 urand4()
{
    return seed = pcg4d(seed);
}

#define rand()  rand4().x
#define rand2() rand4().xy
#define rand3() rand4().xyz
#define rand4() (vec4(urand4()) / 4294967296.)

uvec4 uhash4(vec4 p)  { return pcg4d(floatBitsToUint(p)); }
uvec4 uhash4(float v) { return uhash4(vec4(v)); }
uvec4 uhash4(vec2 v)  { return uhash4(v.xyxy); }
uvec4 uhash4(vec3 v)  { return uhash4(v.xyzx); }

#define hash(v)  hash4(v).x
#define hash2(v) hash4(v).xy
#define hash3(v) hash4(v).xyz
#define hash4(v) (vec4(uhash4(v)) / 4294967296.)

vec2 randOnUnitCircle()
{
    float a = TAU * rand();
    return vec2(cos(a), sin(a));
}

vec2 randInUnitCircle()
{
    vec2 r = rand2();
    r.y *= TAU;
    return sqrt(r.x) * vec2(cos(r.y), sin(r.y));
}

vec3 randOnUnitSphere()
{
    vec2 r = rand2();
    r.x = 2 * r.x - 1;
    r.y *= TAU;
	return vec3(sqrt(1. - r.x * r.x) * vec2(cos(r.y), sin(r.y)), r.x);
}
