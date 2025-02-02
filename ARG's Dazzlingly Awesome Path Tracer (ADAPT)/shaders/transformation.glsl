mat4 rotateX(float a)
{
	return mat4(1,      0,       0, 0, 
		        0, cos(a), -sin(a), 0, 
		        0, sin(a),  cos(a), 0, 
		        0,      0,       0, 1);
}

mat4 rotateY(float a)
{
	return mat4(cos(a), 0, -sin(a), 0, 
		             0, 1,       0, 0,
		        sin(a), 0,  cos(a), 0,
		             0, 0,       0, 1);
}

mat4 rotateZ(float a)
{
	return mat4(cos(a), -sin(a), 0, 0,
		        sin(a),  cos(a), 0, 0,
		             0,       0, 1, 0,
		             0,       0, 0, 1);
}

mat4 translate(vec3 p)
{
	return mat4(1, 0, 0, 0, 
		        0, 1, 0, 0, 
		        0, 0, 1, 0, 
		        p,       1);
}