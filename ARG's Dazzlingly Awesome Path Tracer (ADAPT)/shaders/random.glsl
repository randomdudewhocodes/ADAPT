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

vec2 randOnUnitCircle()
{
    float a = TAU * rand();
    return vec2(cos(a), sin(a));
}

vec3 randOnUnitSphere()
{
    vec2 r = rand2();
    r.x *= TAU;
    r.y = 2 * r.y - 1;
	return vec3(sqrt(1. - r.y * r.y) * vec2(cos(r.x), sin(r.x)), r.y);
}
