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
          IOR, ax, ay,
          opacity;
};

struct hitRecord
{
    float t;
    vec3 p, normal;
    material mat;
    float eta;
};

bool triIntersect(vec3 v0, vec3 v1, vec3 v2, ray r, float tmin, float tmax, inout hitRecord rec)
{
    vec3 v1v0 = v1 - v0;
    vec3 v2v0 = v2 - v0;
    vec3 rov0 = r.o - v0;
    vec3  n = cross( v1v0, v2v0 );
    vec3  q = cross( rov0, r.d );
    float d = 1 / dot( r.d, n );
    float u = d*dot( -q, v2v0 );
    float v = d*dot(  q, v1v0 );
    float t = d*dot( -n, rov0 );

    if(u < 0 || v < 0 || u + v > 1 || t < tmin || t > tmax) return false;

    rec.t = t;
    rec.p = r.o + t * r.d;
    rec.normal = normalize(n);
	return true;
}