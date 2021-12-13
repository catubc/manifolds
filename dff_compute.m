size(temp2)
size(temp,[2])

temp2 = temp(:,:);
for p = 1:(size(temp,[2]))
    temp2(:,p) = temp(:,p)-mean(temp(:,p));
end