% def_rand_dynamics
%
% Defines a 1-dimensional system with random dynamics

% Specify parameters
monomial_degree = 2;
N_terms = 3;

syms x u real

% coeffs = (2*rand(N_coeffs,1) - 1) .* randi([0,1],N_coeffs,1);
coeffs = 2*rand(N_terms,1) - 1;
selectors = randi( [0,1] , N_terms , 2*(monomial_degree + 1) + 4 );

% define monomials of specified degree
x_monomial = sym( ones( 1 , monomial_degree + 1 ) );
u_monomial = sym( ones( 1 , monomial_degree + 1 ) );
for i = 1 : monomial_degree + 1
    x_monomial(1,i) = x_monomial(1,i) * x^(i-1);
    u_monomial(1,i) = u_monomial(1,i) * u^(i-1);
end

% define terms
funcs = [ x_monomial , sin(x) , cos(x) , u_monomial , sin(u) , cos(u) ];
terms = sym( ones(1,N_terms) );
for i = 1 : N_terms
    terms(i) = coeffs(i) * prod( funcs.^selectors(i,:) ); 
end

% dynamics are defined as sum of the terms
xdot = sum( terms );

