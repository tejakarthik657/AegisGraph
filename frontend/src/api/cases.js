import client from "./client";

export const getCases = async (params = {}) =>
  (await client.get("/cases", { params })).data;
export const getCase = async (id) => (await client.get(`/cases/${id}`)).data;
