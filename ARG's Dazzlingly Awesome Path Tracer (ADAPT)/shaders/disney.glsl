#define luminance(v) dot(v, vec3(.3, .6, .1))

void tint(material mat, float eta, out float F0, out vec3 Csheen, out vec3 Cspec)
{
    float lum = luminance(mat.baseColor);
    vec3 tint = lum > 0. ? mat.baseColor / lum : vec3(1);

    F0 = sqr((1. - eta) / (1. + eta));
    
    Cspec  = mix(vec3(1), tint, mat.specularTint) * F0;
    Csheen = mix(vec3(1), tint, mat.sheenTint);
}

vec3 diffuse(material mat, vec3 Csheen, vec3 V, vec3 L, vec3 H, out float pdf)
{
    float LoH = dot(L, H),
          Rr  = 2. * mat.roughness * LoH * LoH,

    // Diffuse
    
    FL     = schlickWeight(L.z),
    FV     = schlickWeight(V.z),
    Fretro = Rr * (FL + FV + FL * FV * (Rr - 1.)),
    Fd     = (1. - .5 * FL) * (1. - .5 * FV),

    // Fake subsurface
    
    Fss90 = .5 * Rr,
    Fss   = mix(1., Fss90, FL) * mix(1., Fss90, FV),
    ss    = 1.25 * (Fss * (1. / (L.z + V.z) - .5) + .5);

    // Sheen
    
    vec3 Fsheen = schlickWeight(LoH) * mat.sheen * Csheen;

    pdf = L.z * INVPI;
    
    return INVPI * mat.baseColor * mix(Fd + Fretro, ss, mat.subsurface) + Fsheen;
}

vec3 reflection(material mat, vec3 V, vec3 L, vec3 H, vec3 F, out float pdf)
{
    float D  = GTR2Aniso(H.z, H.x, H.y, mat.ax, mat.ay),
          G1 = smithGAniso(abs(V.z), V.x, V.y, mat.ax, mat.ay),
          G2 = smithGAniso(abs(L.z), L.x, L.y, mat.ax, mat.ay) * G1;

    pdf = .25 * G1 * D / V.z;
    
    return F * D * G2 / (4. * L.z * V.z);
}

vec3 refraction(material mat, float eta, vec3 V, vec3 L, vec3 H, vec3 F, out float pdf)
{
    float LoH = dot(L, H), VoH = dot(V, H),

    D  = GTR2Aniso(H.z, H.x, H.y, mat.ax, mat.ay),
    G1 = smithGAniso(abs(V.z), V.x, V.y, mat.ax, mat.ay),
    G2 = smithGAniso(abs(L.z), L.x, L.y, mat.ax, mat.ay) * G1;
    
    float jacobian = abs(LoH) / sqr(LoH + VoH * eta);

    pdf = G1 * max(0., VoH) * D * jacobian / V.z;
    
    return sqrt(mat.baseColor) * (1. - F) * D * G2 * abs(VoH) * jacobian * eta * eta / abs(L.z * V.z);
}

float clearcoat(material mat, vec3 V, vec3 L, vec3 H, out float pdf)
{
    float VoH = dot(V, H),

    F = mix(.04, 1., schlickWeight(VoH)),
    D = GTR1(H.z, mat.clearcoatRoughness),
    G = smithG(L.z, .25) * smithG(V.z, .25);

    pdf = .25 * D * H.z / VoH;
    
    return F * D * G;
}

void anisotropicParams(inout hitRecord rec)
{
    float aspect = sqrt(1. - rec.mat.anisotropic * .9);
    rec.mat.ax = rec.mat.roughness / aspect;
    rec.mat.ay = rec.mat.roughness * aspect;
}

vec3 DisneyBSDF(hitRecord rec, vec3 V, vec3 N, vec3 L, out float pdf)
{
    anisotropicParams(rec);
    
    pdf = 0.;
    vec3 f = vec3(0);

    vec3 T, B;
    onb(N, T, B);

    V = toLocal(T, B, N, V);
    L = toLocal(T, B, N, L);

    vec3 H = normalize(L.z > 0. ? L + V : L + V * rec.eta);

    if (H.z < 0.) H = -H;

    // Tint colors
    
    vec3 Csheen, Cspec;
    float F0;
    tint(rec.mat, rec.eta, F0, Csheen, Cspec);

    // Model weights
    
    float dielectricW = (1. - rec.mat.metallic) * (1. - rec.mat.specTrans);
    float metalW      = rec.mat.metallic;
    float glassW      = (1. - rec.mat.metallic) * rec.mat.specTrans;

    // Lobe probabilities
    
    float schlickW = schlickWeight(V.z);

    float diffP       = dielectricW * luminance(rec.mat.baseColor);
    float dielectricP = dielectricW * mix(luminance(Cspec), 1., schlickW);
    float metalP      = metalW * mix(luminance(rec.mat.baseColor), 1., schlickW);
    float glassP      = glassW;
    float clearCoatP  = .25 * rec.mat.clearcoat;

    // Normalize probabilities
    
    float norm = 1. / (diffP + dielectricP + metalP + glassP + clearCoatP);
    
    diffP       *= norm;
    dielectricP *= norm;
    metalP      *= norm;
    glassP      *= norm;
    clearCoatP  *= norm;

    bool reflect = L.z > 0.;

    float tmpPdf = 0.;
    float VoH = abs(dot(V, H));

    // Diffuse
    if (diffP > 0. && reflect)
    {
        f += diffuse(rec.mat, Csheen, V, L, H, tmpPdf) * dielectricW;
        pdf += tmpPdf * diffP;
    }

    // Dielectric Reflection
    if (dielectricP > 0. && reflect)
    {
        float F = (dielectricFresnel(VoH, 1. / rec.eta) - F0) / (1. - F0);

        f += reflection(rec.mat, V, L, H, mix(Cspec, vec3(1), F), tmpPdf) * dielectricW;
        pdf += tmpPdf * dielectricP;
    }

    // Metallic Reflection
    if (metalP > 0.0 && reflect)
    {
        // Tinted to base color
        vec3 F = mix(rec.mat.baseColor, vec3(1), schlickWeight(VoH));

        f += reflection(rec.mat, V, L, H, F, tmpPdf) * metalW;
        pdf += tmpPdf * metalP;
    }

    // Glass/Specular BSDF
    if (glassP > 0.0)
    {
        // Dielectric fresnel (achromatic)
        float F = dielectricFresnel(VoH, rec.eta);

        if (reflect)
        {
            f += reflection(rec.mat, V, L, H, vec3(F), tmpPdf) * glassW;
            pdf += tmpPdf * glassP * F;
        }
        else
        {
            f += refraction(rec.mat, rec.eta, V, L, H, vec3(F), tmpPdf) * glassW;
            pdf += tmpPdf * glassP * (1. - F);
        }
    }

    // Clearcoat
    if (clearCoatP > 0. && reflect)
    {
        f += clearcoat(rec.mat, V, L, H, tmpPdf) * .25 * rec.mat.clearcoat;
        pdf += tmpPdf * clearCoatP;
    }

    return f * abs(L.z);
}

vec3 DisneySample(hitRecord rec, vec3 V, out vec3 L, out float pdf)
{
    anisotropicParams(rec);
    
    pdf = 0.;

    vec3 N = rec.normal, T, B;
    onb(N, T, B);

    V = toLocal(T, B, N, V);

    // Tint colors
    vec3 Csheen, Cspec;
    float F0;
    tint(rec.mat, rec.eta, F0, Csheen, Cspec);

    // Model weights
    float dielectricW = (1. - rec.mat.metallic) * (1. - rec.mat.specTrans);
    float metalW      = rec.mat.metallic;
    float glassW      = (1. - rec.mat.metallic) * rec.mat.specTrans;

    // Lobe probabilities
    float schlick = schlickWeight(V.z);

    float diffP       = dielectricW * luminance(rec.mat.baseColor);
    float dielectricP = dielectricW * mix(luminance(Cspec), 1., schlick);
    float metalP      = metalW * mix(luminance(rec.mat.baseColor), 1., schlick);
    float glassP      = glassW;
    float clearCoatP  = .25 * rec.mat.clearcoat;

    // Normalize probabilities
    float norm = 1. / (diffP + dielectricP + metalP + glassP + clearCoatP);
    diffP *= norm;
    dielectricP *= norm;
    metalP *= norm;
    glassP *= norm;
    clearCoatP *= norm;

    // CDF of the sampling probabilities
    vec3 cdf;
    cdf.x = diffP;
    cdf.y = cdf.x + dielectricP + metalP;
    cdf.z = cdf.y + glassP;

    // Sample a lobe based on its importance
    float r = rand();
    
    if (r < cdf.x) // Diffuse
    {
        L = cosineSample();
    }
    else if (r < cdf.y) // Dielectric + Metallic reflection
    {
        vec3 H = sampleGGXVNDF(V, rec.mat.ax, rec.mat.ay);
        L = reflect(-V, H);
    }
    else if (r < cdf.z) // Glass
    {
        vec3 H = sampleGGXVNDF(V, rec.mat.ax, rec.mat.ay);
        
        float F = dielectricFresnel(abs(dot(V, H)), rec.eta);
        
        L = rand() < F ? reflect(-V, H) : refract(-V, H, rec.eta);
    }
    else // Clearcoat
    {
        vec3 H = sampleGTR1(rec.mat.clearcoatRoughness);

        L = reflect(-V, H);
    }

    L = toWorld(T, B, N, L);
    V = toWorld(T, B, N, V);

    return DisneyBSDF(rec, V, N, L, pdf);
}