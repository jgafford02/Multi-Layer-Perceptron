function [train,test,val] = accuracy(data,output_train,output_test,output_val)
train = 100*sum(output_train == data.training.classes)/data.training.count;
test = 100*sum(output_test == data.test.classes)/data.test_count;
val = 100*sum(output_val == data.validation.classes)/data.validation_count;
end

