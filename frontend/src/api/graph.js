import client from "./client";

export const getGraph = async (id, params = {}) =>
  (await client.get(`/graph/${id}`, { params })).data;
