import { useQuery } from "@tanstack/react-query";

import { loadModel } from "../../../api";

export function useModelQuery() {
  const query = useQuery({
    queryKey: ["sorting-hat-model"],
    queryFn: async () => {
      const result = await loadModel();

      return result;
    },
    staleTime: Infinity,
    gcTime: Infinity,
  });

  return query;
}
