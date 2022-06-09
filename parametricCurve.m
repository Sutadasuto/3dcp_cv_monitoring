function [curvature, x, y] = parametricCurve(curve, t)

segment = 0;

% Find the line line segment containing the value t of interest (t is the
% independent parametric variable)
for s=1:(length(curve.breaks)-1)
   if t >= curve.breaks(s) && t <= curve.breaks(s+1)
       segment = s;
       break
   end
end

if segment == 0
    curvature = nan;
end

% The coefficients correspond to those calculated for the line segment
% containing the value t of interest
x_coefs = curve.coefs(2*s-1, :);
y_coefs = curve.coefs(2*s, :);

% Define the relative value of the independent parametric variable
T = (t-curve.breaks(s));

% Third degree polynomials x(t), y(t)
x = x_coefs(1)*T^3 + x_coefs(2)*T^2 + x_coefs(3)*T + x_coefs(4);
y = y_coefs(1)*T^3 + y_coefs(2)*T^2 + y_coefs(3)*T + y_coefs(4);

% First derivative at t
x_1 = 3*x_coefs(1)*T^2 + 2*x_coefs(2)*T + x_coefs(3);
y_1 = 3*y_coefs(1)*T^2 + 2*y_coefs(2)*T + y_coefs(3);

% Second derivative at t
x_2 = 6*x_coefs(1)*T + 2*x_coefs(2);
y_2 = 6*y_coefs(1)*T + 2*y_coefs(2);

% Curvature calculation at t
k = (x_1*y_2 - y_1*x_2) / (x_1^2 + y_1^2)^1.5;

curvature = [t, k, x, y];
end
