const float TAU    = 6.28318530717958647693,
            PI     = 3.14159265358979323846,
            INVPI  =  .31830988618379067154,
            INV4PI =  .07957747154594766788;

#define sqr(x) (x) * (x)
#define dot2(x) dot(x, x)

float deNaN (float v)
{
    return v != v ? 0. : v;
}

vec3 deNaN(vec3 v)
{
    return vec3(deNaN(v.x), deNaN(v.y), deNaN(v.z));
}