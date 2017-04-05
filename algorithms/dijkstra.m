% TODO: CHECK
function [G_path] = dijkstra(G_edge)
% this function compute minimal path tree from a specific start point by
% using dijkstra's algorithm. The input of this function is a graph
% containing all weighted edges and a start two dimensional point
% cooridnate. The output is also a graph, which contains the minimal
% distance referred to the start point

G_path = Inf(size(G_edge));
G_path(1, 1) = 0;
G_path_copy = G_path;
number_points = size(G_edge, 1);

exposed_set = [];
all_set = 1:size(G_edge, 1);
% eliminate the element from all set, which would never be explored
eliminate_id_list = [];
for i = 1:number_points
    array1 = G_edge(i, :);
    array2 = G_edge(:, i);
    id1 = find(array1 ~= Inf);
    id2 = find(array2 ~= Inf);
    if isempty(id1) && isempty(id2)
        eliminate_id_list = [eliminate_id_list, i];
    end
end
all_set(eliminate_id_list) = [];

while ~isequal(unique(exposed_set), all_set)    % if the exposed set hasn't included all points, then keep exploring
    [min_dis, min_id] = min(G_path_copy(:));
    min_id1 = mod(min_id-1, number_points) + 1;
    min_id2 = ceil(min_id/number_points);
    if isempty(find(exposed_set(:) == min_id1)) || isempty(find(exposed_set(:) == min_id2))        % if the id has not been already exposed
        if isempty(find(exposed_set(:) == min_id1))
            id_new = min_id1;
        else
            id_new = min_id2;
        end
        exposed_set = [exposed_set, id_new];
        
        % compute the candidate distance to fill in
        array1 = G_edge(id_new, :);
        array2 = G_edge(:, id_new);
        id1 = find(array1 ~= Inf);
        id2 = find(array2 ~= Inf);
        dis1 = array1(id1);
        dis2 = array2(id2);
        candidate_list = [];
        candidate_list(:, 1) = [id1'; id2];
        candidate_list(:, 3) = [dis1'; dis2] + min_dis;
        candidate_list(:, 2) = id_new;
        candidate_list = unique(candidate_list, 'rows');
        
        % test each candidate by comparing the current distance in the
        % matrix with this new one and keeping the smallest one
        for i = 1:size(candidate_list, 1)
            row = min(candidate_list(i, [1,2]));
            col = max(candidate_list(i, [1,2]));
            candidate_dis = candidate_list(i, 3);
            if candidate_dis < G_path(row, col)
                G_path(row, col) = candidate_dis;
                G_path_copy(row, col) = candidate_dis;
            end
        end
        G_path_copy(min_id) = Inf;
        %         G_path
    else
        G_path_copy(min_id) = Inf;
        %         G_path
        continue;
    end
end

end