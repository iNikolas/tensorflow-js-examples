import { useQuery } from "@tanstack/react-query";

import { loadModel } from "../../../api";

export function useModelQuery() {
  const query = useQuery({
    queryKey: ["inception-v3-model"],
    queryFn: async () => {
      const result = await loadModel();

      return result;
    },
    staleTime: Infinity,
    gcTime: Infinity,
  });

  return query;
}
