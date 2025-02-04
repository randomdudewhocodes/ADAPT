struct material
{
    vec3 baseColor, emission;
    
    float anisotropic,
          metallic,
          roughness,
          subsurface,
          specularTint,
          sheen,
          sheenTint,
          clearcoat,
          clearcoatRoughness,
          specTrans,
          IOR, ax, ay;
};

struct hitRecord
{
    float t;
    vec3 p, normal;
    material mat;
    float eta;
};

bool hitSphere(vec3 center, float radius, ray r, float tmin, float tmax, inout hitRecord rec)
{
    vec3 oc = r.o - center;
    float b = dot(oc, r.d);
    float h = radius * radius - dot2(oc - b * r.d);

    if (h < 0) return false;

    h = sqrt(h);

	float t1 = -b - h, t2 = -b + h;
	
	float t = t1 < tmin ? t2 : t1;

    if (t > tmin && t < tmax)

    {
        rec.t = t;
        rec.p = r.o + t * r.d;
        rec.normal = (rec.p - center) / radius;
	    return true;
    }
    else return false;
}