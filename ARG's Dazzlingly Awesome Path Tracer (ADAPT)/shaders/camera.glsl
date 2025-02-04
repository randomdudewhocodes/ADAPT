ray getRay(vec3 ro, vec3 lp, vec3 up, float fov, float aperture, float d, vec2 uv)
{
    vec3 ww = normalize(ro - lp),
         uu = normalize(cross(up, ww)),
         vv = cross(ww, uu);
    
    uv *= tan(radians(fov / 2)) * d;

    vec2 rd = aperture / 2 * randInUnitCircle();
    
    vec3 co = ro + uu * rd.x + vv * rd.y;
    
    return ray(co, normalize(ro + uv.x * uu + uv.y * vv - d * ww - co));
}