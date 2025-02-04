
float GTR1(float NoH, float a)
{
    if (a >= 1) return INVPI;
    float a2 = a * a;
    return (a2 - 1) / (PI * log(a2) * (1 + (a2 - 1) * NoH * NoH));
}

vec3 sampleGTR1(float a)
{
    float a2 = a * a;

    float cos2T = (1 - pow(a2, rand())) / (1 - a2);

    return vec3(sqrt(1 - cos2T) * randOnUnitCircle(), sqrt(cos2T));
}

vec3 sampleGGXVNDF(vec3 V, float ax, float ay)
{
    V = normalize(vec3(ax * V.x, ay * V.y, V.z));

    float z = mix(-V.z, 1, rand());

    vec3 H = vec3(sqrt(1 - z * z) * randOnUnitCircle(), z) + V;
    
    return normalize(vec3(ax * H.x, ay * H.y, H.z));
}

float GTR2Aniso(float NoH, float HoX, float HoY, float ax, float ay)
{
    if(ax * ay == 0.) return 1.;

    float a = HoX / ax, b = HoY / ay;
    
    return 1 / (PI * ax * ay * sqr(a * a + b * b + NoH * NoH));
}

float smithG(float NoV, float alphaG)
{
    float a = alphaG * alphaG,
          b = NoV * NoV;
    
    return 2. * NoV / (NoV + sqrt(a + b - a * b));
}

float smithGAniso(float NoV, float VoX, float VoY, float ax, float ay)
{
    if(ax * ay == 0.) return 1.;
    
    float a = VoX * ax,
          b = VoY * ay,
          c = NoV;
    
    return 2. * NoV / (NoV + sqrt(a * a + b * b + c * c));
}

float schlickWeight(float u)
{
    float m = min(1. - u, 1.),
          m2 = m * m;
    
    return m2 * m2 * m;
}

float dielectricFresnel(float cosI, float eta)
{
    float sin2T = eta * eta * (1. - cosI * cosI);

    // Total internal reflection
    
    if (sin2T > 1.) return 1.;

    float cosT = sqrt(1. - sin2T);

    float rs = (eta * cosT - cosI) / (eta * cosT + cosI),
          rp = (eta * cosI - cosT) / (eta * cosI + cosT);

    return .5 * (rs * rs + rp * rp);
}

void onb(vec3 N, out vec3 T, out vec3 B)
{
    B = normalize(cross(N, normalize(N.zxy - dot(N.zxy, N))));
    T = normalize(cross(B, N));
}

vec3 toWorld(vec3 X, vec3 Y, vec3 Z, vec3 V) { return mat3(X, Y, Z) * V; }
vec3 toLocal(vec3 X, vec3 Y, vec3 Z, vec3 V) { return V * mat3(X, Y, Z); }

vec3 cosineSample()
{
    float r = rand();
    return vec3(sqrt(r) * randOnUnitCircle(), sqrt(1. - r));
}