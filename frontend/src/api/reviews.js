import client from "./client";

export const getReviews = async (params = {}) =>
  (await client.get("/reviews", { params })).data;
export const submitReview = async (id, payload) =>
  (await client.post(`/case-management/${id}/reviews`, payload)).data;
