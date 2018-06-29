function output_matrix = class_to_output(v)
n = length(v);  %Number of instances
m = max(v);     %Number of classes

output_matrix = zeros(n,m);
for i = 1:length(v)
    output_matrix(i,v(i)) = 1;
end
end
