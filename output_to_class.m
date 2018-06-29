function classes = output_to_class(outputs)
[values, classes] = max(outputs, [], 2);
end