struct ray
{
    vec3 o, d;
};

vec3 at(ray r, float t)
{
    return r.o + t * r.d;
}